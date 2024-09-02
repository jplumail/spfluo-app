import logging
from typing import TYPE_CHECKING, Optional

from spfluo.utils.array import array_namespace, get_device, to_device
from spfluo.utils.memory import split_batch
from spfluo.utils.volume import interpolate_to_size, pad

if TYPE_CHECKING:
    from spfluo.utils.array import Array, Device

logger = logging.getLogger("spfluo.refinement")


def deconv(
    image: "Array",
    psf: "Array",
    lambda_: "Array",
    device: Optional["Device"] = None,
    batch_size: Optional[int] = None,
):
    """Reconstruct a particule from volumes and their poses.
    M reconstructions can be done at once.

    Args:
        image (Array): stack of N 3D images of shape ((N), D, H, W)
        psf (Array) : 3D image of shape ((N), d, h, w)
        lambda_ (Array): regularization parameters of shape ((N),)
        device (Device): the device to do the computation on.
        batch_size (int or None) : if None, do all the computation at once

    Returns:
        deconv (Array):
            deconvolutions(s) of shape ((N), D, H, W) or ((N), D+1, H+1, W+1)
    """
    logger.info("Calling function deconv")

    xp = array_namespace(image, psf, lambda_)
    host_device = get_device(image)
    compute_device = device
    D, H, W = image.shape[-3:]
    d, h, w = psf.shape[-3:]

    batch = True
    if image.ndim == 3:
        image = image[None]
        batch = False
    if psf.ndim == 3:
        psf = psf[None]
    if lambda_.ndim == 0:
        lambda_ = lambda_[None]
    N = image.shape[0]
    assert psf.shape[0] == N
    assert lambda_.shape[0] == N

    logger.info(
        "Arguments:" f" {N} images of size {D}x{H}x{W}" f" PSF of size {d}x{h}x{w}"
    )

    # pad by 1 pixel to the right if even dimensions
    image = pad(image, ((0, 0), (0, (D + 1) % 2), (0, (H + 1) % 2), (0, (W + 1) % 2)))
    D_, H_, W_ = image.shape[-3:]
    logger.info(f"Reshaped volumes to odd size {D_}x{H_}x{W_}")

    psf = interpolate_to_size(psf, (D_, H_, W_), batch=True)

    num = xp.zeros((N, D_, H_, W_), dtype=xp.complex64, device=host_device)
    den = xp.zeros_like(num, dtype=xp.float32)

    dxyz = xp.zeros((3, 2, 2, 2), device=host_device, dtype=xp.complex64)
    dxyz[0, 0, 0, 0] = 1
    dxyz[0, 1, 0, 0] = -1
    dxyz[1, 0, 0, 0] = 1
    dxyz[1, 0, 1, 0] = -1
    dxyz[2, 0, 0, 0] = 1
    dxyz[2, 0, 0, 1] = -1

    dxyz_padded = pad(
        dxyz, ((0, 0), *[((x - 1) // 2, (x - 2) // 2) for x in (D_, H_, W_)])
    )
    DtD = xp.sum((xp.abs(xp.fft.fftn(dxyz_padded, axes=(1, 2, 3))) ** 2), axis=0)
    den += xp.asarray(lambda_[:, None, None, None] * DtD, dtype=xp.float32)
    del DtD

    # Move data to compute device
    for start, end in split_batch((N,), batch_size):
        images_minibatch = to_device(image[start:end, ...], compute_device)
        psf_minibatch = to_device(psf[start:end, ...], compute_device)

        y = xp.asarray(images_minibatch, dtype=xp.complex64)
        H_ = xp.asarray(psf_minibatch, dtype=xp.complex64)

        H_ = xp.fft.fftn(H_, axes=(-3, -2, -1))

        # Compute numerator
        y = xp.fft.fftn(xp.fft.fftshift(y, axes=(-3, -2, -1)), axes=(-3, -2, -1))
        y = xp.conj(H_) * y
        num[start:end, ...] += to_device(
            xp.sum(y, axis=0, dtype=xp.complex64), host_device
        )  # reduce N dims

        # Compute denominator
        den[start:end, ...] += to_device(
            xp.sum(xp.abs(H_) ** 2, axis=0, dtype=xp.float32), host_device
        )
        del H_

    num = xp.fft.ifftn(num / den, axes=(-3, -2, -1))
    deconv = xp.real(num)
    del num
    deconv = xp.where(deconv > 0, deconv, xp.asarray(0.0, device=host_device))

    deconv = interpolate_to_size(deconv, (D, H, W), batch=True)

    if not batch:
        deconv = deconv[0]

    return deconv
