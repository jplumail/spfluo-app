import os
import pickle
from time import time

import numpy as np
import torch
from scipy import ndimage as ndii
from scipy.signal import correlate
from utils import inverse_affine_transform, pad_to_size, reconstruction_L2
from utils.loading import load_annotations, load_array


def load_views(views_path):
    _, ext = os.path.splitext(views_path)
    if ext == ".csv":  # DONT USE CSV NOT WORKING
        views_ = load_annotations(views_path)
        views = views_[:, 2].astype(int)
        patches_names = np.array(
            [
                os.path.splitext(im_name)[0]
                + "_"
                + patch_index
                + os.path.splitext(im_name)[1]
                for im_name, patch_index in views_[:, [0, 1]]
            ]
        )
    elif ext == ".pickle":
        with open(views_path, "rb") as f:
            views_ = pickle.load(f)
        views = np.concatenate([views_[image_name][0] for image_name in views_.keys()])
        patches_names = np.concatenate(
            [views_[image_name][2] for image_name in views_.keys()]
        )

    return views, patches_names


def load_patches(crop_dir, patches_names):
    patches = [
        load_array(os.path.join(crop_dir, p)) for p in patches_names
    ]  # (N,z,y,x)
    return np.stack(patches, axis=0)


def normalize_patches(patches):
    N = patches.shape[0]
    patch_shape = patches.shape[1:]
    flatten_patches = patches.reshape(N, -1)
    min_patch, max_patch = flatten_patches.min(axis=1), flatten_patches.max(axis=1)
    min_patch = min_patch.reshape(tuple([N] + [1] * len(patch_shape)))
    max_patch = max_patch.reshape(tuple([N] + [1] * len(patch_shape)))
    patches = (patches - min_patch) / (max_patch - min_patch)

    return patches


def polar(image, angles=None, radii=None):
    """Return polar transformed image and log base."""
    shape = image.shape
    center = shape[0] / 2, shape[1] / 2
    if angles is None:
        angles = shape[0]
    if radii is None:
        radii = shape[1]
    theta = np.empty((angles, radii), dtype="float64")
    theta.T[:] = np.linspace(0, 2 * np.pi, angles, endpoint=False)
    # d = radii
    d = np.hypot(shape[0] - center[0], shape[1] - center[1])
    # log_base = 10.0 ** (np.log10(d) / (radii))
    radius = np.empty_like(theta)
    # radius[:] = (
    #    np.power(log_base, np.arange(radii, dtype='float64')) - 1.0
    # )
    radius[:] = np.linspace(0, d, radii)
    x = radius * np.sin(theta) + center[0]
    y = radius * np.cos(theta) + center[1]
    output = np.empty_like(x)
    ndii.map_coordinates(image, [x, y], output=output)
    return output


def get_rotation_angle(im1, im2):
    acf1, acf2 = correlate(im1, im1), correlate(im2, im2)
    polar1 = polar(acf1)
    polar2 = polar(acf2)
    repeated_polar1 = np.concatenate([polar1, polar1], axis=0)
    angular_correlation = correlate(repeated_polar1, polar2, mode="valid")
    x = np.linspace(0, 360, angular_correlation.shape[0])
    best_angle = x[angular_correlation.argmax()]
    return best_angle


def get_translation(im1, im2):
    ccf = correlate(im1, im2, mode="full")
    y, x = np.unravel_index(ccf.argmax(), ccf.shape)
    center = (ccf.shape[0] - 1) / 2, (ccf.shape[1] - 1) / 2
    return ccf[y, x], center[0] - y, center[1] - x


def get_transform(im1, im2):
    angle = get_rotation_angle(im1, im2)
    rotated_patch1 = ndii.rotate(im2, -angle, reshape=False)
    rotated_patch2 = ndii.rotate(im2, -(angle + 180), reshape=False)
    ccf1, ty1, tx1 = get_translation(im1, rotated_patch1)
    ccf2, ty2, tx2 = get_translation(im1, rotated_patch2)
    if ccf1 > ccf2:
        return angle, ty1, tx1
    else:
        return angle + 180, ty2, tx2


polar_dict = {}


def get_polar_special(im):
    global polar_dict
    k = im.tobytes()
    if k in polar_dict:
        return polar_dict[k]
    else:
        p = polar(correlate(im, im))
        polar_dict[k] = p
        return p


def get_rotation_angle_special(im1, im2):
    polar1, polar2 = get_polar_special(im1), get_polar_special(im2)
    repeated_polar1 = np.concatenate([polar1, polar1], axis=0)
    angular_correlation = correlate(repeated_polar1, polar2, mode="valid")
    x = np.linspace(0, 360, angular_correlation.shape[0])
    best_angle = x[angular_correlation.argmax()]
    return best_angle


def get_transform_special(im1, im2):
    angle = get_rotation_angle_special(im1, im2)
    rotated_patch1 = ndii.rotate(im2, -angle, reshape=False)
    rotated_patch2 = ndii.rotate(im2, -(angle + 180), reshape=False)
    ccf1, ty1, tx1 = get_translation(im1, rotated_patch1)
    ccf2, ty2, tx2 = get_translation(im1, rotated_patch2)
    if ccf1 > ccf2:
        return angle, ty1, tx1
    else:
        return angle + 180, ty2, tx2


def transform_patch(patch, R, ty, tx, pad=False):
    if pad:
        h, w = patch.shape
        padw = int(h * (2**0.5 - 1) + 2)
        patch = np.pad(patch, [(padw, padw), (padw, padw)])
    r_patch = ndii.rotate(patch, -R, reshape=False)
    s_patch = ndii.shift(r_patch, np.array([-ty, -tx]))
    if pad:
        s_patch = s_patch[padw:-padw, padw:-padw]
    return s_patch


def build_particle(patches, tree, R, ty, tx):
    r_, ty_, tx_ = tree[3]
    if isinstance(tree[2], list):
        m = tree[0]
        p = patches[tree[1]]
        return (
            transform_patch(
                p + (m - 1) * build_particle(patches, tree[2], r_, ty_, tx_), R, ty, tx
            )
            / m
        )
    else:
        assert tree[0] == 2
        r_, ty_, tx_ = tree[3]
        p = (patches[tree[1]] + transform_patch(patches[tree[2]], r_, ty_, tx_)) / 2
        return transform_patch(p, R, ty, tx)


def fill_params(params, tree, R, ty, tx):
    r_, ty_, tx_ = tree[3]
    params[tree[1]] = (R % 360, ty, tx)
    R_rad = R * np.pi / 180
    if isinstance(tree[2], list):
        fill_params(
            params,
            tree[2],
            R + r_,
            tx * np.sin(R_rad) + ty * np.cos(R_rad) - ty_,
            tx * np.cos(R_rad) - ty * np.sin(R_rad) - tx_,
        )
        # fill_params(params, tree[2], R+r_, ty+ty_, tx+tx_)
    else:
        params[tree[2]] = (
            (R + r_) % 360,
            tx * np.sin(R_rad) + ty * np.cos(R_rad) - ty_,
            tx * np.cos(R_rad) - ty * np.sin(R_rad) - tx_,
        )
        # params[tree[2]] = ((R+r_)%360,  ty+ty_, tx+tx_)


def iterative_alignment(particles, epsilon=1e-2, max_iter=20):
    N, h, w = particles.shape

    pad = int(h * (2**0.5 - 1) + 2)
    # pad = 0
    patches_padded = np.pad(particles, [(0, 0), (pad, pad), (pad, pad)])
    _, H, W = patches_padded.shape

    # Phase A : build an average particle

    patches_param = np.zeros((N, 3))
    order = np.random.choice(np.arange(N), N, replace=False)
    # Center patches
    patches_padded_centered = np.zeros_like(patches_padded)
    for i in range(N):
        y_centermass, x_centermass = ndii.center_of_mass(patches_padded[order[i]])
        center = (H - 1) / 2, (W - 1) / 2
        ty, tx = center[0] - y_centermass, center[1] - x_centermass
        patches_param[i] = [0, -ty, -tx]
        patches_padded_centered[order[i]] = transform_patch(
            patches_padded[order[i]], 0, -ty, -tx
        )

    # Initialize the tree
    tree = [
        2,
        order[0],
        order[1],
        get_transform(patches_padded_centered[order[0]], patches_padded[order[1]]),
    ]

    # Build the tree
    while tree[0] < N:
        m = int(tree[0])
        p_right = build_particle(patches_padded_centered, tree, 0, 0, 0)

        # find the transformation
        p_left = patches_padded_centered[order[m]]
        angle, ty, tx = get_transform(p_left, p_right)

        ## center the particle
        # y_centermass, x_centermass = ndii.center_of_mass(p_right)
        # ty_corr, tx_corr = center[0] - y_centermass, center[1] - x_centermass

        tree = [m + 1, order[m], tree, (angle, ty, tx)]

    # Build the average particle
    avrg_particle = build_particle(patches_padded_centered, tree, 0, 0, 0)
    # plt.imshow(avrg_particle, cmap='gray')
    # plt.show()

    # Get the particles' params from the tree
    patches_param_centered = np.zeros_like(patches_param)
    fill_params(patches_param_centered, tree, 0, 0, 0)
    print(tree)
    # patches_param = patches_param + patches_param_centered

    # Phase B : stabilize the average particle
    order = np.arange(N)
    distance = np.array([3, 3, 3])
    j = 0
    while distance.sum() > epsilon and j < max_iter:
        j += 1
        np.random.shuffle(order)  # shuffle the particles
        new_patches_param = np.array([(0, 0, 0) for i in range(N)])
        last_transforms = [None for i in range(N)]
        for i in range(N):
            # p is the patch to remove
            p = patches_padded_centered[order[i]]
            transformed_p = transform_patch(
                p, *patches_param_centered[order[i]]
            )  # if last_transforms[order[i]] is None else last_transforms[order[i]]
            print(order[i], patches_param_centered[order[i]])

            # plt.figure(figsize=(10,5))
            # plt.subplot(151)
            # plt.imshow(avrg_particle, cmap='gray')
            # remove the patch p from the average
            avrg_particle = (N * avrg_particle - transformed_p) / (N - 1)
            # plt.subplot(152)
            # plt.imshow(p, cmap='gray')
            # plt.subplot(153)
            # plt.imshow(transformed_p, cmap='gray')
            # plt.subplot(154)
            # plt.imshow(avrg_particle, cmap='gray')

            # align the patch p and the new average
            R, ty, tx = get_transform(avrg_particle, p)

            # save the new params of patch p
            new_patches_param[order[i]] = np.array([R, ty, tx])

            # reconstruct the average particle
            new_transformed_p = transform_patch(p, *new_patches_param[order[i]])
            # plt.subplot(155)
            # plt.imshow(new_transformed_p, cmap='gray')
            # plt.show()
            last_transforms[order[i]] = new_transformed_p
            avrg_particle = (new_transformed_p + (N - 1) * avrg_particle) / N

        distances = np.square(
            (new_patches_param - patches_param) / np.array([360, 1, 1])
        )
        distance = distances.mean(axis=0) ** 0.5
        patches_param_centered = new_patches_param

    unpadded_avrg_particle = avrg_particle[pad:-pad, pad:-pad]
    patches_param[:, 0] = patches_param[:, 0] % 360

    variance_map = sum(
        [
            (transform_patch(particles[i], *patches_param[i]) - unpadded_avrg_particle)
            ** 2
            for i in range(N)
        ]
    ) / (N - 1)

    return patches_param, unpadded_avrg_particle, variance_map


def iterative_suppression(patches, n):
    N = patches.shape[0]
    assert n <= N
    n_iter = int(np.ceil(np.log2(N / n)))
    mask_keep = np.ones((N), dtype=bool)
    good_patches = patches.copy()
    for i in range(n_iter):
        params, avrg_particle, variance_map = iterative_alignment(good_patches)
        std = np.array(
            [
                (
                    (transform_patch(good_patches[i], *params[i]) - avrg_particle) ** 2
                ).mean()
                ** 0.5
                for i in range(good_patches.shape[0])
            ]
        )
        if i < n_iter - 1:  # not last iteration
            mask_keep[mask_keep] = std <= np.median(
                std
            )  # we keep half of the particles
        else:
            x = np.where(mask_keep)[0]
            mask_keep[x[np.argsort(std)[n:]]] = False

        good_patches = patches[mask_keep]

    params, avrg_particle, variance_map = iterative_alignment(good_patches)

    return mask_keep, params, avrg_particle, variance_map


def get_first_euler_angle_side(side_particles, top_particles, psf, lambda_=1e-2):
    device = side_particles.device
    dtype = side_particles.dtype

    def as_tensor(x):
        return torch.as_tensor(x, device=device, dtype=dtype)

    t0 = time()
    side_param, _, _ = iterative_alignment(side_particles.sum(dim=1).cpu().numpy())
    t1 = time()
    top_param, _, _ = iterative_alignment(top_particles.sum(dim=1).cpu().numpy())
    t2 = time()

    side_poses = torch.zeros((side_particles.size(0), 6), device=device)
    top_poses = torch.zeros((top_particles.size(0), 6), device=device)

    # Known a priori
    side_poses[:, 1] = 90
    top_poses[:, 1] = 0
    top_poses[:, 0] = 0

    # Found with iterative alignment
    # relative 3rd euler angle of top particles
    # top_poses[:, 2] = as_tensor(top_param[:, 0])
    # relative 1st euler angle of side particles
    side_poses[:, 0] = (-as_tensor(side_param[:, 0])) % 360
    # translations in the XY plane
    side_poses[:, [4, 5]] = as_tensor(side_param[:, [1, 2]])
    top_poses[:, [4, 5]] = as_tensor(top_param[:, [1, 2]])

    # Z-axis translation with center of mass
    side_centers_of_mass = (
        np.array(
            list(map(lambda x: ndii.center_of_mass(x)[0], side_particles.cpu().numpy()))
        )
        - side_particles.size(1) / 2
    )
    side_poses[:, 3] = as_tensor(side_centers_of_mass)
    top_centers_of_mass = (
        np.array(
            list(map(lambda x: ndii.center_of_mass(x)[0], top_particles.cpu().numpy()))
        )
        - side_particles.size(1) / 2
    )
    top_poses[:, 3] = as_tensor(top_centers_of_mass)

    # Find first euler angle of side particles
    step = 1
    delta = torch.arange(0, 360, step)[:, None]
    M = delta.size(0)
    side_poses_expanded = side_poses[None].repeat(M, 1, 1)
    side_poses_expanded[:, :, 0] += as_tensor(delta)
    top_poses_expanded = top_poses[None].repeat(M, 1, 1)
    poses_expanded = torch.cat((side_poses_expanded, top_poses_expanded), dim=1)
    particles = torch.cat((side_particles, top_particles), dim=0)

    lambda_ = as_tensor(lambda_)
    t3 = time()
    recon, den = reconstruction_L2(
        particles, psf, poses_expanded, lambda_[None].repeat(M)
    )
    # dt = [0 for _ in range(7)]
    t4 = time()

    MSE = torch.empty((recon.size(0),))
    recon_fft = torch.fft.fftn(torch.fft.fftshift(recon, dim=(1, 2, 3)), dim=(1, 2, 3))
    # recon_fft = torch.fft.fftn(recon, dim=(1,2,3))
    for i in range(recon.size(0)):
        patches_transformed = inverse_affine_transform(
            particles[:, None], poses_expanded[i]
        )[:, 0]
        patches_transformed_fft = torch.fft.fftn(
            torch.fft.fftshift(patches_transformed, dim=(1, 2, 3)), dim=(1, 2, 3)
        )
        # patches_transformed_fft = torch.fft.fftn(
        #    patches_transformed,
        #    dim=(1,2,3)
        # )
        poses_no_tvec = poses_expanded[i]
        poses_no_tvec[:, 3:] = 0
        psf_transformed = inverse_affine_transform(
            psf.repeat(poses_expanded.size(1), 1, 1, 1, 1), poses_no_tvec
        )[:, 0]

        psf_transformed_fft = torch.fft.fftn(
            torch.fft.fftshift(
                pad_to_size(
                    psf_transformed, psf_transformed.shape[:1] + particles.shape[-3:]
                ),
                dim=(1, 2, 3),
            ),
            dim=(1, 2, 3),
        )
        # psf_transformed_fft = torch.fft.fftn(
        #    pad_to_size(
        #       psf_transformed, psf_transformed.shape[:1]+particles.shape[-3:]
        #    ),
        #    dim=(1,2,3)
        # )
        recons_conv_fft = recon_fft[i] * psf_transformed_fft

        MSE[i] = (
            recons_conv_fft - patches_transformed_fft
        ).square().sum() / particles.size(0)

    best_i = MSE.argmin()
    best_recon = recon[best_i]
    side_poses[:, 0] = side_poses_expanded[best_i, :, 0]
    t5 = time()

    print(f"1st iterative alignment : {t1-t0:.2f}s")
    print(f"2nd iterative alignment : {t2-t1:.2f}s")
    print(f"create poses : {t3-t2:.2f}s")
    print(f"L2 Reconstructions : {t4-t3:.2f}s")
    print(f"MSEs Computations : {t5-t4:.2f}s")

    return side_poses, top_poses, best_recon, MSE[best_i]
