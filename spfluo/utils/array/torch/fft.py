from torch import fft as torch_fft
from torch.fft import *  # noqa: F403


def fftn(x, s=None, axes=None, norm="backward", dim=None, out=None):
    if dim is not None:
        axes = dim
    if out is not None:
        return torch_fft.fftn(x, s=s, dim=axes, norm=norm, out=out)
    return torch_fft.fftn(x, s=s, dim=axes, norm=norm)


def ifftn(x, s=None, axes=None, norm="backward", dim=None, out=None):
    if dim is not None:
        axes = dim
    if out is not None:
        return torch_fft.ifftn(x, s=s, dim=axes, norm=norm, out=out)
    return torch_fft.ifftn(x, s=s, dim=axes, norm=norm)


def fftshift(x, /, *, axes=None, norm="backward", dim=None):
    if dim is not None:
        axes = dim
    return torch_fft.fftshift(x, dim=axes)
