import math
from typing import Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
import scipy.ndimage as ndii
import utils_memory
from mpl_toolkits.axes_grid1 import make_axes_locatable


def affine_transform(input: torch.Tensor, matrix: torch.Tensor, offset=0.0, output_shape=None, output=None, order=1, mode='zeros', cval=0.0, prefilter=True) -> torch.Tensor:
    """Rotate the volume according to the transform matrix.
    Matches the `scipy.ndimage.affine_transform` function at best with the
    `torch.nn.functional.grid_sample` function

    Args:
        input (torch.Tensor): 3D images of shape (N, C, D, H, W)
        matrix (torch.Tensor): transform matrices of shape (N, 3), (N,3,3), (N,4,4) or (N,3,4).
        offset (float or torch.Tensor): offset of the grid.
        output_shape (tuple): shape of the output.
        output: not implemented
        order (int): must be 1. Only linear interpolation is implemented.
        mode (str): How to fill blank space. 'zeros', 'border' or 'reflection'
        cval (float): not implemented
        prefilter (bool): not implemented
    
    Returns:
        torch.Tensor: Rotated volumes of shape (N, C, D, H, W)
    """
    N, C, D, H, W = input.size()
    tensor_kwargs = dict(device=input.device, dtype=input.dtype)

    if type(offset) == float:
        tvec = torch.tensor([offset, offset, offset], **tensor_kwargs)[None]
    elif offset.shape == (3,):
        tvec = torch.as_tensor(offset, **tensor_kwargs)[None]
    elif offset.shape == (N, 3):
        tvec = torch.as_tensor(offset, **tensor_kwargs)
    else:
        raise ValueError("Offset should be a float, a sequence of size 3 or a tensor of size (N,3).")
        
    if matrix.size() == torch.Size([N, 3, 3]):
        rotMat = matrix
    elif matrix.size() == torch.Size([N, 3]):
        rotMat = torch.stack([torch.diag(matrix[i]) for i in range(N)])
    elif matrix.size() == torch.Size([N, 4, 4]) or matrix.size() == torch.Size([N, 3, 4]):
        rotMat = matrix[:, :3, :3]
        tvec = matrix[:, :3, 3]
    else:
        raise ValueError(f"Matrix should be a tensor of shape {(N,3)}, {(N,3,3)}, {(N,4,4)} or {(N,3,4)}. Found matrix of shape {matrix.size()}")

    if output_shape is None:
        output_shape = (D, H, W)

    if output is not None:
        raise NotImplementedError()
    
    if order > 1:
        raise NotImplementedError()

    pytorch_modes = ['zeros', 'border', 'reflection']
    if mode not in pytorch_modes:
        raise NotImplementedError(f'Only {pytorch_modes} are available')

    if cval != 0:
        raise NotImplementedError()

    if not prefilter:
        raise NotImplementedError()
    
    return _affine_transform(input, rotMat, tvec, output_shape, mode, tensor_kwargs)


def _affine_transform(input, rotMat, tvec, output_shape, mode, tensor_kwargs):
    output_shape = list(output_shape)
    rotated_vol = input.new_empty(list(input.shape[:2])+output_shape)
    grid = torch.stack(torch.meshgrid([torch.linspace(0, d-1, steps=d, **tensor_kwargs) for d in output_shape], indexing='ij'), dim=-1)
    c = torch.tensor([0 for d in output_shape], **tensor_kwargs)
    input_shape = torch.as_tensor(input.shape[2:], **tensor_kwargs)
    for start, end in utils_memory.split_batch_func("affine_transform", input, rotMat):
        grid_batch = (rotMat[start:end,None,None,None] @ ((grid-c)[None,...,None]))[...,0] + c + tvec[start:end,None,None,None,:]
        grid_batch = - 1 + 1 / input_shape + 2 * grid_batch / input_shape
        rotated_vol[start:end] = F.grid_sample(input[start:end], grid_batch[:,:,:,:,[2,1,0]], mode='bilinear', align_corners=False, padding_mode=mode)
    rotated_vol[:,:,:output_shape[0],:output_shape[1],:output_shape[2]]
    return rotated_vol


def reconstruction_L2(volumes: torch.Tensor, psf: torch.Tensor, poses: torch.Tensor, lambda_: torch.Tensor) -> torch.Tensor:
    """Reconstruct a particule from volumes and their poses. Multiple reconstructions can be done at once.

    Args:
        volumes (torch.Tensor): stack(s) of N 3D images of shape (..., N, D, H, W)
        psf (torch.Tensor) : 3D image(s) of shape (..., d, h, w)
        poses (torch.Tensor): stack(s) of N transform matrices of shape (..., N, 6).
            Euler angles in the 'zxz' convention in degrees and translation vector tz, ty, tx
        lambda_ (torch.Tensor): regularization parameters of shape (...)
    
    Returns:
        recon (torch.Tensor): reconstruction(s) of shape (..., D, H, W)
        den (torch.Tensor): something of shape (..., D, H, W)
    """
    N, D, H, W = volumes.size(-4), volumes.size(-3), volumes.size(-2), volumes.size(-1)
    d, h, w = psf.size(-3), psf.size(-2), psf.size(-1)
    batch_dims = np.array(volumes.shape[:-4])
    size_batch = int(batch_dims.prod())
    volumes_rotated = inverse_affine_transform(volumes.view(size_batch*N,1,D,H,W), poses.view(size_batch*N,6))[:,0].view(*batch_dims,N,D,H,W)
    psfs = psf.unsqueeze(-4)
    psfs = psfs.repeat(*tuple([1]*len(batch_dims)), N, 1, 1, 1)
    psfs_rotated = inverse_affine_transform(psfs.view(size_batch*N,1,d,h,w), poses.view(size_batch*N,6))[:,0].view(*batch_dims,N,d,h,w)
    psfs_rotated_padded = pad_to_size(psfs_rotated, volumes_rotated.size())

    H = torch.fft.fftn(psfs_rotated_padded, dim=(-3,-2,-1))
    y = torch.fft.fftn(torch.fft.fftshift(volumes_rotated, dim=(-3,-2,-1)), dim=(-3,-2,-1))

    HtH = (H.conj() * H).abs().mean(dim=-4)
    Hty = (H.conj() * y).mean(dim=-4)
                
    dxyz = torch.zeros((3,2,2,2), device=volumes.device)
    dxyz[0,0,0,0] = 1 
    dxyz[0,1,0,0] = -1
    dxyz[1,0,0,0] = 1 
    dxyz[1,0,1,0] = -1
    dxyz[2,0,0,0] = 1 
    dxyz[2,0,0,1] = -1

    dxyz_padded = pad_to_size(dxyz, (3,) + volumes.shape[-3:])
    DtD = (torch.fft.fftn(dxyz_padded).abs()**2).sum(dim=0)

    den = HtH + lambda_[...,None,None,None] * DtD
    recon = torch.fft.ifftn(Hty/den, dim=(-3,-2,-1)).real
    recon = torch.clamp(recon, min=0)

    return recon, den


def fftn(x: torch.Tensor, dim: Tuple[int]=None, out=None) -> torch.Tensor:
    """Computes N dimensional FFT of x in batch. Tries to avoid out-of-memory errors.

    Args:
        x: data
        dim: tuple of size N, dimensions where FFTs will be computed
        out: the output tensor
    Returns:
        y: data in the Fourier domain, shape of x 
    """
    if dim is None:
        return torch.fft.fftn(x, out=out)
    else:
        if x.is_complex():
            dtype = x.dtype
        else:
            if x.dtype is torch.float32:
                dtype = torch.complex64
            elif x.dtype is torch.float64:
                dtype = torch.complex128
        batch_indices = torch.ones((x.ndim,), dtype=bool)
        batch_indices[list(dim)] = False
        batch_indices = batch_indices.nonzero()[:,0]
        batch_slices = [slice(None,None) for i in range(x.ndim)]

        if out is not None:
            y = out
        else:
            y = x.new_empty(size=x.size(), dtype=dtype)
        for batch_idx in utils_memory.split_batch_func("fftn", x, dim):
            if type(batch_idx) is tuple: batch_idx = [batch_idx]
            for d, (start, end) in zip(batch_indices, batch_idx):
                batch_slices[d] = slice(start, end)
            torch.fft.fftn(x[tuple(batch_slices)], dim=dim, out=y[tuple(batch_slices)])
        return y


def ifftn(x: torch.Tensor, dim: Tuple[int]=None, out=None) -> torch.Tensor:
    """Computes N dimensional inverse FFT of x in batch. Tries to avoid out-of-memory errors.

    Args:
        x: data
        dim: tuple of size N, dimensions where FFTs will be computed
        out: the output tensor
    Returns:
        y: data in the Fourier domain, shape of x 
    """
    if dim is None or len(dim)==x.ndim:
        return torch.fft.ifftn(x, out=out)
    else:
        if x.is_complex():
            dtype = x.dtype
        else:
            if x.dtype is torch.float32:
                dtype = torch.complex64
            elif x.dtype is torch.float64:
                dtype = torch.complex128
        batch_indices = torch.ones((x.ndim,), dtype=bool)
        batch_indices[list(dim)] = False
        batch_indices = batch_indices.nonzero()[:,0]
        batch_slices = [slice(None,None) for i in range(x.ndim)]

        if out is not None:
            y = out
        else:
            y = x.new_empty(size=x.size(), dtype=dtype)
        for batch_idx in utils_memory.split_batch_func("fftn", x, dim):
            if type(batch_idx) is tuple: batch_idx = [batch_idx]
            for d, (start, end) in zip(batch_indices, batch_idx):
                batch_slices[d] = slice(start, end)
            torch.fft.ifftn(x[tuple(batch_slices)], dim=dim, out=y[tuple(batch_slices)])
        return y



def pad_to_size(volume: torch.Tensor, output_size: torch.Size) -> torch.Tensor:
    output_size = torch.as_tensor(output_size)
    pad_size = torch.ceil((output_size - torch.as_tensor(volume.size()))/2)

    padding = tuple(np.asarray([[max(pad_size[i],0), max(pad_size[i],0)] for i in range(len(pad_size))], dtype=int).flatten()[::-1].tolist())
    output_volume = F.pad(volume, padding)

    shift = (torch.as_tensor(output_volume.size()) - output_size) / 2
    slices = [
            slice(int(np.ceil(shift[i])),-int(np.floor(shift[i]))) if shift[i] > 0 and np.floor(shift[i]) > 0
            else slice(int(np.ceil(shift[i])),None) if shift[i] > 0
            else slice(None, None)
            for i in range(len(shift))
        ]

    return output_volume[tuple(slices)]


def fourier_shift(volume_freq: torch.Tensor, shift: torch.Tensor, nb_spatial_dims=None):
    """
    Args:
        volume (torch.Tensor): volume in the Fourier domain ({...}, [...])
            where [...] corresponds to the N spatial dimensions and {...} corresponds to the batched dimensions
        shift (torch.Tensor): shift to apply to the volume ({{...}}, N)
            where {{...}} corresponds to batched dimensions. {...} and {{...}} must be broadcastable.
        nb_spatial_dims (int): number of spatial dimensions N
    Returns:
        out (torch.Tensor): volume shifted in the Fourier domain
    """
    tensor_kwargs = {'device': volume_freq.device}
    if volume_freq.dtype == torch.complex128:
        tensor_kwargs['dtype'] = torch.float64
    elif volume_freq.dtype == torch.complex64:
        tensor_kwargs['dtype'] = torch.float32
    elif volume_freq.dtype == torch.complex32:
        tensor_kwargs['dtype'] = torch.float16
    else:
        print("Volume must be complex")
    if nb_spatial_dims is None:
        nb_spatial_dims = volume_freq.ndim
    spatial_shape = torch.as_tensor(volume_freq.size()[-nb_spatial_dims:], **tensor_kwargs)
    assert shift.size(-1) == nb_spatial_dims
    shift = shift.view(*shift.shape[:-1],*[1 for _ in range(nb_spatial_dims)],-1)
    
    grid_freq = torch.stack(    
        torch.meshgrid(
            *[torch.fft.fftfreq(int(s), **tensor_kwargs) for s in spatial_shape],
            indexing='ij'
        ), dim=-1
    )
    phase_shift = (grid_freq * shift).sum(-1)
        
    # Fourier shift
    out = volume_freq * torch.exp(-1j*2*torch.pi*phase_shift)

    return out


    




def hann_window(shape: Tuple[int], **kwargs) -> torch.Tensor:
    """Computes N dimensional Hann window.

    Args:
        shape: shape of the final window
        kwargs: keyword arguments for torch.hann_window function
    Returns:
         Hann window of the shape asked
    """
    windows = [torch.hann_window(s, **kwargs) for s in shape]
    view = [1] * len(shape)
    hw = torch.ones(shape, device=windows[0].device, dtype=windows[0].dtype)
    for i in range(len(windows)):
        view_ = list(view)
        view_[i] = -1
        hw *= windows[i].view(tuple(view_))
    return hw


def convolution_matching_poses_grid(
    reference: torch.Tensor,
    volumes: torch.Tensor,
    psf: torch.Tensor,
    potential_poses: torch.Tensor
) -> Tuple[torch.Tensor]:
    """Find the best pose from a list of poses for each volume
    Params:
        reference (torch.Tensor) : reference 3D image of shape (D, H, W)
        volumes (torch.Tensor) : volumes to match of shape (N, D, H, W)
        psf (torch.Tensor): 3D PSF of shape (D, H, W)
        potential_poses (torch.Tensor): poses to test of shape (M, 6)
    Returns:
        best_poses (torch.Tensor): best poses for each volume of shape (N, 6)
        best_errors (torch.Tensor): dftRegistration error associated to each pose (N,)
    """
    # Shapes
    M, _ = potential_poses.shape
    N, D, H, W = volumes.shape

    # PSF
    h = torch.fft.fftn(torch.fft.fftshift(pad_to_size(psf, (D,H,W))))

    shifts = torch.empty((N,M,3))
    errors = torch.empty((N,M))
    for (start1, end1), (start2, end2) in utils_memory.split_batch_func(
        "convolution_matching_poses_grid", reference, volumes, psf, potential_poses
    ):
        potential_poses_minibatch = potential_poses[start2:end2]

        # Volumes to frequency space
        volumes_freq = torch.fft.fftn(volumes[start1:end1], dim=(1,2,3))

        # Rotate the reference
        reference_minibatch = reference.repeat(end2-start2, 1, 1, 1)
        reference_minibatch = affine_transform(reference_minibatch[:,None], potential_poses_minibatch)[:,0]
        reference_minibatch = h * torch.fft.fftn(reference_minibatch, dim=(1,2,3))

        # Registration
        err, sh = dftregistrationND(reference_minibatch[None], volumes_freq[:,None], nb_spatial_dims=3)
        sh = torch.stack(list(sh), dim=-1)
        
        errors[start1:end1, start2:end2] = err
        shifts[start1:end1, start2:end2] = sh
        
        del volumes_freq, reference_minibatch, err, sh
        torch.cuda.empty_cache()
    
    best_errors, best_indices = torch.min(errors, dim=1)
    best_poses = potential_poses[best_indices]
    best_poses[:, 3:] = shifts[np.arange(N), best_indices]
    
    return best_poses, best_errors


def convolution_matching_poses_refined(
    reference: torch.Tensor,
    volumes: torch.Tensor,
    psf: torch.Tensor,
    potential_poses: torch.Tensor
) -> Tuple[torch.Tensor]:
    """Find the best pose from a list of poses for each volume. There can be a different list of pose
    for each volume.
    Params:
        reference (torch.Tensor) : reference 3D image of shape (D, H, W)
        volumes (torch.Tensor) : volumes to match of shape (N, D, H, W)
        psf (torch.Tensor): 3D PSF of shape (D, H, W)
        potential_poses (torch.Tensor): poses to test of shape (N, M, 6)
    Returns:
        best_poses (torch.Tensor): best poses for each volume of shape (N, 6)
        best_errors (torch.Tensor): dftRegistration error associated to each pose (N,)
    """
    # Shapes
    N1, M, d = potential_poses.shape
    N, D, H, W = volumes.shape
    assert N==N1

    # PSF
    h = torch.fft.fftn(torch.fft.fftshift(pad_to_size(psf, (D,H,W))))

    shifts = torch.empty((N,M,3))
    errors = torch.empty((N,M))
    for (start1, end1), (start2, end2) in utils_memory.split_batch_func(
        "convolution_matching_poses_refined", reference, volumes, psf, potential_poses
    ):
        minibatch_size = (end1-start1)*(end2-start2)
        potential_poses_minibatch = potential_poses[start1:end1, start2:end2].contiguous()

        # Volumes to Fourier space
        volumes_freq = torch.fft.fftn(volumes[start1:end1], dim=(1,2,3))

        # Rotate the reference
        reference_minibatch = reference.repeat(minibatch_size, 1, 1, 1)
        reference_minibatch = affine_transform(reference_minibatch[:,None], potential_poses_minibatch.view(minibatch_size,d))[:,0].view(end1-start1,end2-start2,D,H,W)
        reference_minibatch = h * torch.fft.fftn(reference_minibatch, dim=(2,3,4))

        # Registration
        err, sh = dftregistrationND(reference_minibatch, volumes_freq[:,None], nb_spatial_dims=3)
        sh = torch.stack(list(sh), dim=-1)

        errors[start1:end1, start2:end2] = err
        shifts[start1:end1, start2:end2] = sh

        del volumes_freq, reference_minibatch, err, sh
        torch.cuda.empty_cache()
    
    errors, best_indices = torch.min(errors, dim=1)
    best_poses = potential_poses[np.arange(N), best_indices]
    best_poses[:, 3:] = shifts[np.arange(N), best_indices]
    return best_poses, errors


def _upsampled_dft(data, upsampled_region_size, upsample_factor=1, axis_offsets=None, nb_spatial_dims=3):
    tensor_kwargs = {'device': data.device, 'dtype': None}
    upsampled_region_size = [upsampled_region_size, ] * nb_spatial_dims
    dim_properties = list(zip(data.shape[-nb_spatial_dims:], upsampled_region_size, axis_offsets.permute(-1, *tuple(range(axis_offsets.ndim-1)))))
    im2pi = 1j * 2 * np.pi
    for (n_items, ups_size, ax_offset) in dim_properties[::-1]:
        kernel = ((torch.arange(ups_size, **tensor_kwargs) - ax_offset[...,None])[..., None]
                  * torch.fft.fftfreq(n_items, upsample_factor, **tensor_kwargs))
        kernel = torch.exp(-im2pi * kernel)
        kernel = kernel.type(data.dtype)
        data = torch.einsum(
            kernel, [..., 0, nb_spatial_dims],
            data, [...]+list(range(1,1+nb_spatial_dims)),
            [..., 0]+list(range(1,1+nb_spatial_dims-1))
        )
    return data


def unravel_index(x, dims):
    one = torch.tensor([1], dtype=dims.dtype, device=dims.device)
    a = torch.cat((one, dims.flip([0])[:-1]))
    dim_prod = torch.cumprod(a, dim=0).flip([0]).type(torch.float)
    return torch.floor(x[...,None]/dim_prod) % dims


def cross_correlation_max(x: torch.Tensor, y: torch.Tensor, normalization: str, nb_spatial_dims: int=None) -> Tuple[torch.Tensor]:
    """ Compute cross-correlation between x and y
    Params:
        x (torch.Tensor) of shape (B, ...) where (...) corresponds to the N spatial dimensions
        y (torch.Tensor) of the same shape
    Returns:
        maxi (torch.Tensor): cross correlatio maximum of shape (B,)
        shift (Tuple[torch.Tensor]): tuple of N tensors of size (B,)
        image_product (torch.Tensor): product of size (B, ...)
    """
    nb_spatial_dims = nb_spatial_dims if nb_spatial_dims is not None else x.ndim
    output_shape = torch.as_tensor(torch.broadcast_shapes(x.size(), y.size()), dtype=torch.int64, device=x.device)
    spatial_dims = list(range(len(output_shape)-nb_spatial_dims, len(output_shape)))
    spatial_shape = output_shape[-nb_spatial_dims:]
    z = x * y.conj()
    if normalization == "phase":
        eps = torch.finfo(z.real.dtype).eps
        z /= torch.max(z.abs(), torch.as_tensor(100*eps))
    cc = ifftn(z, dim=spatial_dims)
    cc = torch.mul(cc, cc.conj(), out=cc).real
    cc = torch.flatten(cc, start_dim=-nb_spatial_dims)
    maxi, max_idx = torch.max(cc, dim=-1)
    shift = unravel_index(max_idx.type(torch.int64), spatial_shape)
    return maxi, shift, z


def dftregistrationND(reference: torch.Tensor, moving_images: torch.Tensor, nb_spatial_dims: int=None, upsample_factor: int=1, normalization: str="phase") -> torch.Tensor:
    """ Phase cross-correlation between a reference and moving_images
    Params:
        reference (torch.Tensor): image of shape ({...}, [...]) where [...] corresponds to the N spatial dimensions
        moving_images (torch.Tensor): images to register of shape ({{...}}, [...]). {...} and {{...}} are broadcasted to (...)
        nb_spatial_dims (int): specify the N spatial dimensions
        upsample_factor (float): upsampling factor. Images will be registered up to 1/upsample_factor.
    Returns:
        error (torch.Tensor): tensor of shape (...)
        shift (Tuple[torch.Tensor]): tuple of N tensors of size (...)
    """
    device = reference.device
    output_shape = torch.as_tensor(torch.broadcast_shapes(reference.size(), moving_images.size()))
    if nb_spatial_dims is None:
        nb_spatial_dims = len(output_shape)
    spatial_dims = list(range(len(output_shape)-nb_spatial_dims, len(output_shape)))
    other_dims = list(range(0, len(output_shape)-nb_spatial_dims))
    spatial_shapes = output_shape[spatial_dims]
    other_shapes = output_shape[other_dims]
    midpoints = torch.tensor([torch.fix(axis_size / 2) for axis_size in spatial_shapes], device=device)
    
    # Single pixel registration
    error, shift, image_product = cross_correlation_max(reference, moving_images, normalization, nb_spatial_dims=nb_spatial_dims)
        
    # Now change shifts so that they represent relative shifts and not indices
    spatial_shapes_broadcasted = torch.broadcast_to(spatial_shapes, shift.size()).to(device)
    shift[shift > midpoints] -= spatial_shapes_broadcasted[shift > midpoints]

    spatial_size = torch.prod(spatial_shapes).type(reference.dtype)

    if upsample_factor == 1:
        rg00 = torch.sum((reference * reference.conj()), dim=tuple(range(reference.ndim-nb_spatial_dims,reference.ndim))) / spatial_size
        rf00 = torch.sum((moving_images * moving_images.conj()), dim=tuple(range(moving_images.ndim-nb_spatial_dims,moving_images.ndim))) / spatial_size
    else:
        upsample_factor = torch.tensor(upsample_factor, device=device, dtype=torch.float)
        shift = torch.round(shift * upsample_factor) / upsample_factor
        upsampled_region_size = torch.ceil(upsample_factor * 1.5)
        dftshift = torch.fix(upsampled_region_size / 2.0)
        sample_region_offset = dftshift - shift * upsample_factor
        cross_correlation = _upsampled_dft(image_product.conj(),
                                           upsampled_region_size,
                                           upsample_factor,
                                           sample_region_offset,
                                           nb_spatial_dims).conj()
        cross_correlation = (cross_correlation * cross_correlation.conj()).real
        error, max_idx = torch.max(cross_correlation.reshape(*tuple(other_shapes), -1), dim=-1)
        maxima = unravel_index(max_idx, torch.as_tensor(cross_correlation.shape[-nb_spatial_dims:], device=max_idx.device))
        maxima -= dftshift

        shift += maxima / upsample_factor

        rg00 = torch.sum((reference * reference.conj()), dim=tuple(range(reference.ndim-nb_spatial_dims,reference.ndim)))
        rf00 = torch.sum((moving_images * moving_images.conj()), dim=tuple(range(moving_images.ndim-nb_spatial_dims,moving_images.ndim)))

    error = torch.tensor([1.0], device=device) - error / (rg00.real * rf00.real)
    error = torch.sqrt(error.abs())    

    return error, tuple([shift[...,i] for i in range(shift.size(-1))])


def discretize_sphere_uniformly(N, dtype=torch.float64, device=None):
    ''' Generates a list of the two first euler angles that describe a uniform discretization of the sphere with the Fibonnaci sphere algorithm
    :param N: number of points
    '''
    epsilon = 0.5
    goldenRatio = (1 + 5 ** 0.5) / 2
    i = torch.arange(0, N, dtype=dtype, device=device)
    theta = torch.remainder(2 * torch.pi * i / goldenRatio, 2*torch.pi)
    phi = torch.acos(1 - 2 * (i + epsilon) / N)
    precision = 360 / (torch.pi * N) ** 0.5
    return theta*180/torch.pi, phi*180/torch.pi, precision


def find_L(precision):
    return math.ceil(((360 / precision) ** 2) / torch.pi)


def create_poses_grid(L, dtype=torch.float64, device=None):
    theta, phi, precision = discretize_sphere_uniformly(L, dtype, device)
    list_angles = torch.stack([torch.zeros_like(theta), phi, theta], dim=-1)
    M = list_angles.shape[0]
    list_translation = torch.zeros((M,3), dtype=dtype, device=device)
    potential_poses = torch.cat([list_angles,list_translation], dim=1)
    return potential_poses, precision


def find_angles_grid(reconstruction, patches, psf, precision=10):
    L = find_L(precision)
    potential_poses, _ = create_poses_grid(L, reconstruction.dtype, reconstruction.device)

    best_poses, best_errors = convolution_matching_poses_grid(reconstruction, patches, psf, potential_poses)

    return best_poses, best_errors


def get_refined_values1D(loc, N, sigma=10, range=0.8, device=None, dtype=torch.float32):
    d = torch.distributions.normal.Normal(loc, sigma)
    step = range / N
    lowest = 0.5 - torch.floor(N/2) * step
    highest = 0.5 + torch.ceil(N/2) * step
    return d.icdf(torch.arange(lowest, highest, step, device=device, dtype=dtype))


def get_refined_valuesND(locs, N, sigmas, ranges, **kwargs):
    n = len(locs)
    values_1d = [get_refined_values1D(locs[i], int(N**(1/n)), sigma=sigmas[i], range=ranges[i], **kwargs) for i in range(n)]

    return torch.cartesian_prod(*values_1d)


def create_refined_poses(poses, M, **kwargs):
    potential_poses = torch.clone(poses).unsqueeze(1).repeat(1,M,1)
    for i in range(poses.size(0)):
        pose = poses[i]
        refined_angles = get_refined_valuesND(poses[[1,2]], M, [5,5], [0.9,0.9], **kwargs)
        potential_poses[i, :, [1,2]] = refined_angles
    
    return potential_poses


def refine_poses(reconstruction, patches, psf, guessed_poses, M=20):
    device = reconstruction.device
    dtype = reconstruction.dtype
    potential_poses = create_refined_poses(guessed_poses, M, dtype=dtype, device=device)

    best_poses, best_errors = convolution_matching_poses_refined(reconstruction, patches, psf, potential_poses, max_batch=8)

    return best_poses, best_errors


def distance_poses(p1, p2):
    """ Compute the rotation distance and the euclidean distance between p1 and p2.
    Parameters:
        p1, p2 : torch.Tensor of shape (..., 6). Must be broadcastable. Represents poses (theta,psi,gamma,tz,ty,tx).
    Returns:
        distances : Tuple[torch.Tensor] of shape broadcasted dims.
    """
    # Rotation distance
    rot1, rot2 = p1[...,:3], p2[...,:3]
    rot1 = Rotation.from_euler("ZXZ", rot1.cpu().numpy().reshape(-1,3), degrees=True)
    rot2 = Rotation.from_euler("ZXZ", rot2.cpu().numpy().reshape(-1,3), degrees=True)
    points1 = rot1.apply(np.array([0,0,1]), inverse=False).reshape(p1.shape[:-1]+(3,))
    points2 = rot2.apply(np.array([0,0,1]), inverse=False).reshape(p2.shape[:-1]+(3,))
    angle = np.arccos((points1 * points2).sum(axis=-1)) * 180 / np.pi
    angle = torch.as_tensor(angle, dtype=p1.dtype, device=p1.device)

    # Euclidian distance
    t1, t2 = p1[...,3:], p2[...,3:]
    d = ((t1 - t2)**2).sum(dim=-1) ** 0.5

    return angle, d


def first_reconstruction(patches, views, poses, psf, step=10):
    errors = []
    recons= []
    lambda_ = 5e-2
    poses_known = torch.zeros_like(poses)
    poses_known[views==0, 1:3] = 0
    poses_known[views==1, 1] = 90
    deltas = torch.arange(0, 360, step, dtype=patches.dtype)
    for delta in deltas:
        poses_known[views==1, 2] = poses[views==1, 2] + delta

        # reconstruction L2
        mask_top_side = torch.logical_or(torch.as_tensor(views==0), torch.as_tensor(views==1))
        recon_noised, _ = reconstruction_L2(patches[mask_top_side], psf, poses_known[mask_top_side], lambda_)
        recons.append(recon_noised)
        
        # compute error
        N = patches[mask_top_side].shape[0]
        recon_noised_transformed = affine_transform(recon_noised[None,None].repeat(N,1,1,1,1), poses_known[mask_top_side])[:,0]
        error = (((recon_noised_transformed - patches[mask_top_side])**2).view(N,-1).sum(dim=1)**0.5).sum() / N
        errors.append(error)

    recons = torch.stack(recons)
    errors = torch.stack(errors)
    i = errors.argmin()

    return deltas[i], recons[i], errors[i]


def normalize_patches(patches: torch.Tensor) -> torch.Tensor:
    """Normalize N patches by computing min/max for each patch
    Params: patches (torch.Tensor) of shape (N, ...)
    Returns: normalized_patches (torch.Tensor) of shape (N, ...)
    """
    N = patches.size(0)
    patch_shape = patches.shape[1:]
    flatten_patches = patches.view(N, -1)
    min_patch, _ = flatten_patches.min(dim=1)
    max_patch, _ = flatten_patches.max(dim=1)
    min_patch = min_patch.view(tuple([N]+[1]*len(patch_shape)))
    max_patch = max_patch.view(tuple([N]+[1]*len(patch_shape)))
    normalized_patches = (patches - min_patch) / (max_patch - min_patch)
    
    return normalized_patches


def disp3D(fig, *ims, axis_off=False):
    axes = fig.subplots(1, len(ims))
    if len(ims) == 1:
        axes = [axes]
    if axis_off:
        for ax in axes: ax.set_axis_off()
    for i in range(len(ims)):
        views = [
            ims[i][ims[i].shape[0]//2,:,:],
            ims[i][:,ims[i].shape[1]//2,:],
            ims[i][:,:,ims[i].shape[2]//2]
        ]
        axes[i].set_aspect(1.)
        #views = [normalize_patches(torch.from_numpy(v)).cpu().numpy() for v in views]

        divider = make_axes_locatable(axes[i])
        # below height and pad are in inches
        
        ax_x = divider.append_axes("right", size=f'{100*ims[i].shape[0]/ims[i].shape[2]}%', pad='5%', sharex=axes[i])
        ax_y = divider.append_axes("bottom",size=f'{100*ims[i].shape[0]/ims[i].shape[1]}%', pad='5%', sharey=axes[i])

        # make some labels invisible
        axes[i].xaxis.set_tick_params(labeltop=True, top=True, labelbottom=False, bottom=False)
        ax_x.yaxis.set_tick_params(labelleft=False, left=False, right=True)
        ax_x.xaxis.set_tick_params(top=True, labeltop=True, bottom=True, labelbottom=False)
        ax_y.xaxis.set_tick_params(bottom=True, labelbottom=False, top=False)
        ax_y.yaxis.set_tick_params(right=True)

        # show slice info
        axes[i].text(0, 2, f"Z={ims[i].shape[0]//2}", color='white', bbox=dict(boxstyle='square'))
        ax_x.text(0, 2, f"Y={ims[i].shape[1]//2}", color='white', bbox=dict(boxstyle='square'))
        ax_y.text(0, 2, f"X={ims[i].shape[2]//2}", color='white', bbox=dict(boxstyle='square'))

        axes[i].imshow(views[0], cmap='gray')
        ax_x.imshow(ndii.rotate(views[1],90)[::-1], cmap='gray')
        ax_y.imshow(views[2], cmap='gray')

def disp2D(fig, *ims, **imshowkwargs):
    h = int(np.floor(len(ims)**0.5))
    w = int(np.ceil(len(ims)/h))
    axes = fig.subplots(h, w)
    if type(axes) == np.ndarray:
        axes = axes.flatten()
        for ax in axes: ax.set_axis_off()
        for i in range(len(ims)):
            axes[i].imshow(ims[i], **imshowkwargs)
    else:
        axes.set_axis_off()
        axes.imshow(ims[0], **imshowkwargs)