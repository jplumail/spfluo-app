import functools
from typing import TYPE_CHECKING, TypeAlias

from numpy.array_api._typing import Array as ArrayAPIArray
from numpy.array_api._typing import Device as ArrayAPIDevice
from numpy.array_api._typing import Dtype as ArrayAPIDtype

import spfluo
from spfluo._vendored.array_api_compat import (
    array_namespace,
    is_array_api_obj,
    numpy,
    to_device,
)

libs = [numpy]
if spfluo.has_torch:
    from spfluo._vendored.array_api_compat import torch

    libs.append(torch)
else:
    torch = None

if spfluo.has_cupy:
    from spfluo._vendored.array_api_compat import cupy

    libs.append(cupy)
else:
    cupy = None

Array: TypeAlias = ArrayAPIArray
Device: TypeAlias = ArrayAPIDevice
Dtype: TypeAlias = ArrayAPIDtype

if TYPE_CHECKING:
    from spfluo._vendored.array_api_compat.common._helpers import array_api_module


def numpy_only_compatibility(numpy_func):
    """
    Apply this decorator to numpy only functions to make them compliant
    with the array-api
    numpy_func: Callable
        signature (*args, **kwargs) -> array like object
    """

    @functools.wraps(numpy_func)
    def func(*args, **kwargs) -> Array:
        array_args = list(filter(is_array_api_obj, args))
        array_kwargs = list(filter(is_array_api_obj, kwargs.values()))
        xp = array_namespace(*array_args, *array_kwargs)
        numpy_args = []
        for arg in args:
            arg_ = arg
            if is_array_api_obj(arg):
                arg_ = xp.asarray(arg)
                arg_ = xp.to_device(arg_, "cpu")
            numpy_args.append(arg_)
        numpy_kwargs = {}
        for k, arg in kwargs.items():
            arg_ = arg
            if is_array_api_obj(arg):
                arg_ = xp.asarray(arg)
                arg_ = xp.to_device(arg_, "cpu")
            numpy_kwargs[k] = arg_

        devices = set(map(xp.device, array_args + array_kwargs))
        if len(devices) != 1:
            raise TypeError(f"Multiple devices found in args: {devices}")
        (device,) = devices

        return xp.asarray(numpy_func(*numpy_args, **numpy_kwargs), device=device)

    return func


def to_numpy(*xs) -> numpy.ndarray | tuple[numpy.ndarray]:
    ret = tuple([numpy.asarray(to_device(x, "cpu")) for x in xs])
    if len(xs) == 1:
        return ret[0]
    else:
        return ret


def get_namespace_device(
    xp: "array_api_module | None" = None,  # type: ignore
    device: "Device | None" = None,
    gpu: bool | None = None,
) -> "tuple[array_api_module, Device]":  # type: ignore
    if xp is not None:
        if device is not None:
            return xp, device
        if xp == torch:
            if gpu is None:
                device = "cuda" if spfluo.has_torch_cuda else "cpu"
            elif gpu:
                device = "cuda"
            else:
                device = "cpu"
        elif xp == cupy:
            if gpu is False:
                raise RuntimeError(f"{xp} cannot create non cuda arrays")
            device = None
        elif xp == numpy:
            if gpu is True:
                raise RuntimeError(f"{xp} cannot create gpu arrays")
            device = "cpu"
        else:
            raise RuntimeError(
                f"{xp} not supported. Must be one of {(torch, cupy, numpy)}"
            )
    else:
        if device is not None:
            raise RuntimeError("device provided but xp not provided.")
        if gpu is None:
            if spfluo.has_torch_cuda:
                xp = torch
                device = "cuda"
            elif spfluo.has_cupy:
                xp = cupy
                device = None
            elif spfluo.has_torch:
                xp = torch
                device = "cpu"
            else:
                xp = numpy
                device = "cpu"
        elif gpu:
            if spfluo.has_torch_cuda:
                xp = torch
                device = "cuda"
            elif spfluo.has_cupy:
                xp = cupy
                device = None
            else:
                raise RuntimeError("GPU asked but no backend found")
        else:
            if spfluo.has_torch:
                xp = torch
                device = "cpu"
            else:
                xp = numpy
                device = "cpu"
    return xp, device


__all__ = [Array, array_namespace, is_array_api_obj, to_device, *libs, to_numpy]

if TYPE_CHECKING:
    __all__ = [
        Array,
        array_namespace,
        is_array_api_obj,
        to_device,
        *libs,
        to_numpy,
        array_api_module,
    ]
