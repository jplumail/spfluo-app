import torch
import numpy as np
import spfluo.utils
from spfluo.utils import dftregistrationND
from skimage import data, util
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift


def dftregistration(reference, image, nb_spatial_dims=None, upsample_factor=1, normalization="phase", device='cpu'):
    error, shift = dftregistrationND(
        torch.as_tensor(reference, device=device),
        torch.as_tensor(image, device=device),
        nb_spatial_dims,
        upsample_factor,
        normalization
    )
    error = error.cpu().numpy()
    shift = np.stack([shift[i].cpu().numpy() for i in range(len(shift))], axis=-1)

    return shift, error


def test_simple_dftregistration():
    """
    Test of 1 dftRegistration in 2D
    """
    image = data.camera().astype(float)
    shift = (-22.4, 13.32)
    # The shift corresponds to the pixel offset relative to the reference image
    image_fourier = np.fft.fftn(image)
    offset_image = fourier_shift(image_fourier, shift)
    offset_image = np.fft.ifftn(offset_image)
    
    shift1, error1 = dftregistration(image_fourier, offset_image, normalization=None)
    shift2, error2, _ = phase_cross_correlation(image_fourier, offset_image, space='fourier', overlap_ratio=0, normalization=None)

    eps = 1e-3
    assert ((shift1 - shift2) < eps).all()
    assert (error1 - error2) < eps


def test_simple_phasenorm_dftregistration():
    """
    Test of 1 dftRegistration in 2D
    """
    image = data.camera().astype(float)
    shift = (-22.4, 13.32)
    # The shift corresponds to the pixel offset relative to the reference image
    image_fourier = np.fft.fftn(image)
    offset_image = fourier_shift(image_fourier, shift)
    offset_image = np.fft.ifftn(offset_image)
    
    shift1, error1 = dftregistration(image_fourier, offset_image, normalization="phase")
    shift2, error2, _ = phase_cross_correlation(image_fourier, offset_image, space='fourier', overlap_ratio=0, normalization="phase")

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
    image_fourier = np.fft.fftn(image)
    offset_images = np.stack([fourier_shift(image_fourier, shift) for shift in shifts], axis=0)

    shift1, error1 = dftregistration(image_fourier, offset_images, nb_spatial_dims=2, normalization=None)
    shift2, error2, _ = zip(*[phase_cross_correlation(image_fourier, offset_image, space='fourier', overlap_ratio=0, normalization=None) for offset_image in offset_images])
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
    image_fourier = np.fft.fftn(util.img_as_float(data.cells3d()[:, 1, :, :]))
    shift = np.random.randn(d) * 10
    offset_image = fourier_shift(image_fourier, shift)

    shift1, error1 = dftregistration(image_fourier, offset_image, nb_spatial_dims=d, normalization=None)
    shift2, error2, _ = phase_cross_correlation(image_fourier, offset_image, space='fourier', overlap_ratio=1, normalization=None)

    eps = 1e-3
    assert (np.abs(-shift2 - shift) <= 1).all()
    assert (np.abs(shift1 - shift2) < eps).all()
    assert (np.abs(error1 - error2) < eps).all()


def test_batch_3d_dftregistration():
    """
    Test of batch of 5 dftRegistration in 3D
    """
    d = 3
    N = 5
    image_fourier = np.fft.fftn(util.img_as_float(data.cells3d()[:, 1, :, :]))
    shifts = np.random.randn(N,d) * 100
    offset_images = np.stack([fourier_shift(image_fourier, shift) for shift in shifts], axis=0)

    shift1, error1 = dftregistration(image_fourier, offset_images, nb_spatial_dims=d, normalization=None)
    shift2, error2, _ = zip(*[phase_cross_correlation(image_fourier, offset_image, space='fourier', overlap_ratio=0, normalization=None) for offset_image in offset_images])
    shift2 = np.stack(shift2, axis=0)
    error2 = np.array(error2)

    eps = 1e-3
    assert (np.abs(shift1 - shift2) < eps).all()
    assert (np.abs(error1 - error2) < eps).all()


def test_broadcasting_dftregistration():
    """
    Test of broadcasted dftRegistration in 2D. (N,H,W) with (M,N,H,W)
    """
    N = 2
    M = 3
    image = data.camera().astype(float)
    reference_shifts = np.random.randn(N,2) * 6
    reference_images = np.stack([fourier_shift(np.fft.fftn(image), shift) for shift in reference_shifts], axis=0) # size N,H,W

    offset_shifts = np.random.randn(M,2) * 6
    offset_images = np.stack([np.stack([fourier_shift(np.fft.fftn(ref_image), shift) for shift in offset_shifts], axis=0) for ref_image in reference_images], axis=1) # size M,N,H,W

    shift1, error1 = dftregistration(reference_images, offset_images, nb_spatial_dims=2, normalization=None)
    res = np.stack([np.stack([phase_cross_correlation(ref_image, offset_image, space='fourier', overlap_ratio=0, normalization=None) for offset_image in offset_images[:,i]],axis=0) for i, ref_image in enumerate(reference_images)], axis=1)
    shift2 = np.stack(res[..., 0].flatten()).reshape(M,N,2)
    error2 = res[..., 1]

    eps = 1e-3
    assert (np.abs(shift1 - shift2) < eps).all()
    assert (np.abs(error1 - error2) < eps).all()


def test_broadcasting3d_dftregistration():
    """
    Test of broadcasted dftRegistration in 3D. (N,D,H,W) with (M,D,N,H,W)
    """
    N = 2
    M = 3
    image = util.img_as_float(data.cells3d()[:, 1, :, :])
    reference_shifts = np.random.randn(N, 3) * 6
    reference_images = np.stack([fourier_shift(np.fft.fftn(image), shift) for shift in reference_shifts], axis=0) # size N,D,H,W

    offset_shifts = np.random.randn(M,3) * 6
    offset_images = np.stack([np.stack([fourier_shift(np.fft.fftn(ref_image), shift) for shift in offset_shifts], axis=0) for ref_image in reference_images], axis=1) # size M,N,D,H,W

    shift1, error1 = dftregistration(reference_images, offset_images, nb_spatial_dims=3, normalization=None)
    res = np.stack([np.stack([phase_cross_correlation(ref_image, offset_image, space='fourier', overlap_ratio=0, normalization=None) for offset_image in offset_images[:,i]],axis=0) for i, ref_image in enumerate(reference_images)], axis=1)
    shift2 = np.stack(res[..., 0].flatten()).reshape(M,N,3)
    error2 = res[..., 1]

    eps = 1e-3
    assert (np.abs(shift1 - shift2) < eps).all()
    assert (np.abs(error1 - error2) < eps).all()


def test_upsample_dftregistration():
    """
    Test of 1 dftRegistration in 2D with upsampling factor > 1
    """
    image = data.camera().astype(float)
    shift = (-22.4, 13.32)
    # The shift corresponds to the pixel offset relative to the reference image
    image_fourier = np.fft.fftn(image)
    offset_image = fourier_shift(image_fourier, shift)
    
    shift1, error1 = dftregistration(image_fourier, offset_image, upsample_factor=100, normalization=None)
    shift2, error2, _ = phase_cross_correlation(image_fourier, offset_image, space='fourier', upsample_factor=100, overlap_ratio=0, normalization=None)

    eps = 1e-3
    assert ((shift1 - shift2) < eps).all()
    assert (error1 - error2) < eps


def test_upsample_broadcasting3d_dftregistration():
    """
    Test of broadcasted dftRegistration in 3D with upsampling>1. (N,D,H,W) with (M,D,N,H,W)
    """
    N = 2
    M = 3
    image = util.img_as_float(data.cells3d()[:, 1, :, :])
    reference_shifts = np.random.randn(N, 3) * 6
    reference_images = np.stack([fourier_shift(np.fft.fftn(image), shift) for shift in reference_shifts], axis=0) # size N,D,H,W

    offset_shifts = np.random.randn(M,3) * 6
    offset_images = np.stack([np.stack([fourier_shift(np.fft.fftn(ref_image), shift) for shift in offset_shifts], axis=0) for ref_image in reference_images], axis=1) # size M,N,D,H,W

    shift1, error1 = dftregistration(reference_images, offset_images, nb_spatial_dims=3, upsample_factor=10, normalization=None)
    res = np.stack([np.stack([phase_cross_correlation(ref_image, offset_image, space='fourier', upsample_factor=10, overlap_ratio=0.3, normalization=None) for offset_image in offset_images[:,i]],axis=0) for i, ref_image in enumerate(reference_images)], axis=1)
    shift2 = np.stack(res[..., 0].flatten()).reshape(M,N,3)
    error2 = res[..., 1]

    eps = 1e-3
    assert (np.abs(shift1 - shift2) < eps).all()
    assert (np.abs(error1 - error2) < eps).all()


def test_upsample_cuda_dftregistration():
    """
    Test of 1 dftRegistration in 2D with upsampling factor > 1
    """
    image = data.camera().astype(float)
    shift = (-22.4, 13.32)
    # The shift corresponds to the pixel offset relative to the reference image
    image = np.fft.fftn(image)
    offset_image = fourier_shift(image, shift)
    
    shift1, error1 = dftregistration(image, offset_image, upsample_factor=10, device='cuda', normalization=None)
    shift2, error2, _ = phase_cross_correlation(image, offset_image, space='fourier', upsample_factor=10, overlap_ratio=0, normalization=None)

    eps = 1e-3
    assert ((shift1 - shift2) < eps).all()
    assert (error1 - error2) < eps


####################################################
# Test affine_transform against the scipy function #
####################################################

from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation as R

def affine_transform_gpu(vol, mat, offset=0.0, output_shape=None, device='cpu'):
    out = spfluo.utils.affine_transform(
        torch.as_tensor(vol, device=device),
        torch.as_tensor(mat, device=device),
        offset=offset,
        output_shape=output_shape
    )
    return out.cpu().numpy()

def create_2d_rot_mat(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

def is_affine_close(im1, im2):
    """Because scipy's and pytorch interpolations at borders don't behave equivalently, we add a margin"""
    D, H, W = im1.shape
    return np.isclose(im1, im2).sum() > (D*W*H-2*(H*D+D*W+W*H))

def test_affine_transform_simple():
    N = 10
    image = util.img_as_float(data.cells3d()[:, 1, :, :])
    matrices = np.empty((N, 4, 4))
    for i in range(N):
        matrices[i] = np.eye(4)
        matrices[i, :3,:3] = np.eye(3) + R.random().as_matrix()*0.1
        matrices[i, :3, 3] = np.random.randn(3)
    
    out_scipy = [affine_transform(image, m, order=1) for m in matrices]
    out_pt = list(affine_transform_gpu(np.stack([image]*N)[:,None], matrices)[:,0])
    assert all([is_affine_close(x, y) for x, y in zip(out_scipy, out_pt)])

def test_affine_transform_output_shape():
    output_shapes = [(64,32,32), (32,32,32), (64,128,256), (128,128,57), (57,56,55)]
    image = util.img_as_float(data.cells3d()[:, 1, :, :])
    matrix = np.eye(4)
    matrix[:3,:3] = np.eye(3) + R.random().as_matrix()*0.1
    matrix[:3, 3] = np.random.randn(3)
    
    out_scipy = [affine_transform(image, matrix, order=1, output_shape=o) for o in output_shapes]
    out_pt = [affine_transform_gpu(image[None,None], matrix[None], output_shape=o)[:,0] for o in output_shapes]
    assert all([is_affine_close(x, y) for x, y in zip(out_scipy, out_pt)])

def test_affine_transform_offset():
    N = 10
    image = util.img_as_float(data.cells3d()[:, 1, :, :])
    matrix = np.eye(3) + R.random().as_matrix()*0.1
    matrices = np.stack([matrix]*N)
    offsets = np.random.randn(N, 3)
    
    out_scipy = [affine_transform(image, matrix, order=1, offset=o) for o in offsets]
    out_pt = affine_transform_gpu(np.stack([image]*N)[:,None], matrices, offset=offsets)
    assert all([is_affine_close(x, y) for x, y in zip(out_scipy, out_pt)])

#################################################
# Test fourier_shift against the scipy function #
#################################################

def fourier_shift2(volume_freq, shift, nb_spatial_dims=None, device='cpu'):
    out = spfluo.utils.fourier_shift(
        torch.as_tensor(volume_freq, device=device),
        torch.as_tensor(shift, device=device),
        nb_spatial_dims
    )

    return out.cpu().numpy()


def test_simple_fourier_shift():
    """
    Test of 1 fourier shift in 2D
    """
    image = data.camera().astype(float)
    shift = (-22.4, 13.32)
    # The shift corresponds to the pixel offset relative to the reference image
    offset_image1 = fourier_shift(np.fft.fftn(image), shift)
    offset_image2 = fourier_shift2(np.fft.fftn(image), shift)

    assert np.isclose(offset_image1, offset_image2).all()


def test_broadcasting_fourier_shift():
    """
    Test broadcasted fourier shift in 2D
    """
    M = 2
    N = 10
    images = np.stack([data.camera().astype(float) for i in range(M)])
    images[0, :10, :10] = 0. # images[0] and images[1] are different
    shifts = np.random.randn(N, 2)
    # The shift corresponds to the pixel offset relative to the reference image
    images_fft = np.fft.fftn(images, axes=(1,2))
    offset_images1 = np.stack([np.stack([fourier_shift(image_fft, shift) for shift in shifts]) for image_fft in images_fft])
    offset_images2 = fourier_shift2(images_fft[:,None], shifts[None,:], nb_spatial_dims=2)

    assert np.isclose(offset_images1, offset_images2).all()


#######################
# Test distance_poses #
#######################

def test_distance_poses():
    p1 = torch.tensor([90, 90, 0, 1, 0, 0]).type(torch.float)
    p2 = torch.tensor([-90, 90, 0, 0, 1, 0]).type(torch.float)
    angle, t = spfluo.utils.distance_poses(p1, p2)

    assert torch.isclose(angle, torch.tensor(180.))
    assert torch.isclose(t, torch.tensor(2.)**0.5)

if __name__ == '__main__':
    test_batch_dftregistration()