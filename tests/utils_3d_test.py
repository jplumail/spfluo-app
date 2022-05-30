import torch
import numpy as np
import utils_3d
from utils_3d import convolution_matching_poses_refined, dftregistrationND, convolution_matching_poses_grid, find_angles_grid, reconstruction_L2
from skimage import data, util
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift


def dftregistration(reference, image, nb_spatial_dims=None, device='cpu'):
    error, shift = dftregistrationND(
        torch.as_tensor(reference, device=device),
        torch.as_tensor(image, device=device),
        nb_spatial_dims
    )
    error = error.cpu().numpy()
    shift = np.stack([shift[i].cpu().numpy() for i in range(len(shift))], axis=-1)

    return shift, error


##################################################################################
# Test dftregistration against the scikit-image function phase_cross_correlation #
##################################################################################

def test_simple_dftregistration():
    """
    Test of 1 dftRegistration in 2D
    """
    image = data.camera().astype(float)
    shift = (-22.4, 13.32)
    # The shift corresponds to the pixel offset relative to the reference image
    offset_image = fourier_shift(np.fft.fftn(image), shift)
    offset_image = np.fft.ifftn(offset_image)
    
    shift1, error1 = dftregistration(image, offset_image)
    shift2, error2, _ = phase_cross_correlation(image, offset_image, space='fourier', overlap_ratio=0, normalization=None)

    eps = 1e-3
    assert ((shift1 - shift2) < eps).all()
    assert (error1 - error2) < eps


def test_batch_dftregistration():
    """
    Test of a batch of 5 dftRegistration in 2D
    """
    N = 5
    image = data.camera().astype(float)
    shifts = np.random.randn(N,2) * 6
    offset_images = np.stack([fourier_shift(np.fft.fftn(image), shift) for shift in shifts], axis=0)

    shift1, error1 = dftregistration(image, offset_images, nb_spatial_dims=2)
    shift2, error2, _ = zip(*[phase_cross_correlation(image, offset_image, space='fourier', overlap_ratio=0, normalization=None) for offset_image in offset_images])
    shift2 = np.stack(shift2, axis=0)
    error2 = np.array(error2)

    eps = 1e-3
    assert ((shift1 - shift2) < eps).all()
    assert ((error1 - error2) < eps).all()


def test_3d_dftregistration():
    """
    Test of 1 dftRegistration in 3D
    """
    d = 3
    image = util.img_as_float(data.cells3d()[:, 1, :, :])
    shift = np.random.randn(d) * 100
    offset_image = fourier_shift(np.fft.fftn(image), shift)
    offset_image = np.fft.ifftn(offset_image)

    shift1, error1 = dftregistration(image, offset_image, nb_spatial_dims=d)
    shift2, error2, _ = phase_cross_correlation(image, offset_image, space='fourier', overlap_ratio=0, normalization=None)

    eps = 1e-3
    assert ((shift1 - shift2) < eps).all()
    assert ((error1 - error2) < eps).all()


def test_batch_3d_dftregistration():
    """
    Test of batch of 5 dftRegistration in 3D
    """
    d = 3
    N = 5
    image = util.img_as_float(data.cells3d()[:, 1, :, :])
    shifts = np.random.randn(N,d) * 100
    offset_images = np.stack([fourier_shift(np.fft.fftn(image), shift) for shift in shifts], axis=0)

    shift1, error1 = dftregistration(image, offset_images, nb_spatial_dims=d)
    shift2, error2, _ = zip(*[phase_cross_correlation(image, offset_image, space='fourier', overlap_ratio=0, normalization=None) for offset_image in offset_images])
    shift2 = np.stack(shift2, axis=0)
    error2 = np.array(error2)

    eps = 1e-3
    assert ((shift1 - shift2) < eps).all()
    assert ((error1 - error2) < eps).all()


def test_broadcasting_dftregistration():
    """
    Test of broadcasted dftRegistration in 2D. (N,H,W) with (M,N,H,W)
    """
    N = 10
    M = 20
    image = data.camera().astype(float)
    reference_shifts = np.random.randn(N,2) * 6
    reference_images = np.stack([fourier_shift(np.fft.fftn(image), shift) for shift in reference_shifts], axis=0) # size N,H,W

    offset_shifts = np.random.randn(M,2) * 6
    offset_images = np.stack([np.stack([fourier_shift(np.fft.fftn(ref_image), shift) for shift in offset_shifts], axis=0) for ref_image in reference_images], axis=1) # size M,N,H,W

    shift1, error1 = dftregistration(reference_images, offset_images, nb_spatial_dims=2)
    res = np.stack([np.stack([phase_cross_correlation(ref_image, offset_image, space='fourier', overlap_ratio=0, normalization=None) for offset_image in offset_images[:,i]],axis=0) for i, ref_image in enumerate(reference_images)], axis=1)
    shift2 = np.stack(res[..., 0].flatten()).reshape(M,N,2)
    error2 = res[..., 1]

    eps = 1e-3
    assert ((shift1 - shift2) < eps).all()
    assert ((error1 - error2) < eps).all()


##########################
# Test reconstruction_L2 #
##########################

def test_shapes_reconstruction_L2():
    N, D, H, W = 100, 32, 32, 32
    volumes = torch.randn((N, D, H, W))
    psf = torch.randn((D, H, W))
    poses = torch.randn((N, 6))
    lambda_ = torch.tensor(1.)
    recon, den = reconstruction_L2(volumes, psf, poses, lambda_)
    
    assert recon.shape == (D, H, W)
    assert den.shape == (D, H, W)


def test_parallel_reconstruction_L2():
    batch_dims = (5,5,)
    N, D, H, W = 100, 32, 32, 32
    volumes = torch.randn(batch_dims+(N, D, H, W))
    psf = torch.randn(batch_dims+(D, H, W))
    poses = torch.randn(batch_dims+(N, 6))
    lambda_ = torch.randn(batch_dims)
    recon, _ = reconstruction_L2(volumes, psf, poses, lambda_)
    recon2 = torch.stack([torch.stack([reconstruction_L2(vv, pp, ppoo, ll)[0] for vv, pp, ppoo, ll in zip(v,p,po,l)]) for v, p, po, l in zip(volumes, psf, poses, lambda_)])

    assert recon.shape == batch_dims+(D, H, W)
    assert torch.isclose(recon, recon2).all()

#############################
# Test convolution_matching #
#############################

def test_memory_convolution_matching_poses_grid():
    device = 'cuda'
    D = 32
    for N in [1, 10, 1000, 1500, 5000]:
        N = int(N)
        M = int(10000 / N**0.5)
        reference = torch.randn((D, D, D), device=device)
        volumes = torch.randn((N, D, D, D), device=device)
        psf = torch.randn((D, D, D), device=device)
        potential_poses = torch.randn((M, 6), device=device)

        best_poses, errors = convolution_matching_poses_grid(reference, volumes, psf, potential_poses)

        assert best_poses.shape == (N, 6)
        assert errors.shape == (N,)


def test_shapes_convolution_matching_poses_grid():
    M, d = 5, 6
    N, D, H, W = 100, 32, 32, 32
    reference = torch.randn((D, H, W))
    volumes = torch.randn((N, D, H, W))
    psf = torch.randn((D, H, W))
    potential_poses = torch.randn((M, d))

    best_poses, errors = convolution_matching_poses_grid(reference, volumes, psf, potential_poses)

    assert best_poses.shape == (N, d)
    assert errors.shape == (N,)


def test_shapes_convolution_matching_poses_refined():
    M, d = 5, 6
    N, D, H, W = 100, 32, 32, 32
    reference = torch.randn((D, H, W))
    volumes = torch.randn((N, D, H, W))
    psf = torch.randn((D, H, W))
    potential_poses = torch.randn((N, M, d))

    best_poses, errors = convolution_matching_poses_refined(reference, volumes, psf, potential_poses)

    assert best_poses.shape == (N, d)
    assert errors.shape == (N,)


###################
# Test find_angle #
###################

def test_shapes_find_angles_grid():
    if torch.cuda.is_available(): device = 'cuda'
    else: device = 'cpu'

    N, D, H, W = 150, 32, 32, 32
    reconstruction = torch.randn((D, H, W), device=device)
    patches = torch.randn((N, D, H, W), device=device)
    psf = torch.randn((D, H, W), device=device)

    best_poses, errors = find_angles_grid(reconstruction, patches, psf, precision=10)
    
    assert best_poses.shape == (N, 6)
    assert errors.shape == (N,)


#######################
# Test distance_poses #
#######################

def test_distance_poses():
    p1 = torch.tensor([0, 90, 90, 1, 0, 0]).type(torch.float)
    p2 = torch.tensor([0, 90, -90, 0, 1, 0]).type(torch.float)
    angle, t = utils_3d.distance_poses(p1, p2)

    assert torch.isclose(angle, torch.tensor(180.))
    assert torch.isclose(t, torch.tensor(2.)**0.5)

if __name__ == '__main__':
    test_memory_convolution_matching_poses_grid()