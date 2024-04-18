import functools
from typing import TYPE_CHECKING, Any, TypeAlias, overload

from array_api_compat import (
    array_namespace as _array_namespace,
)
from array_api_compat import (
    device as get_device,
)
from array_api_compat import (
    is_array_api_obj,
    is_cupy_array,
    is_numpy_array,
    is_torch_array,
    numpy,
    to_device,
)

import spfluo


def array_namespace(*xs, api_version=None, _use_compat=True) -> "array_api_module":
    """
    Get the array API compatible namespace for the arrays `xs`.

    `xs` should contain one or more arrays.

    Typical usage is

        def your_function(x, y):
            xp = array_api_compat.array_namespace(x, y)
            # Now use xp as the array library namespace
            return xp.mean(x, axis=0) + 2*xp.std(y, axis=0)

    api_version should be the newest version of the spec that you need support
    for (currently the compat library wrapped APIs only support v2021.12).
    """
    xp: array_api_module = _array_namespace(
        *xs, api_version=api_version, _use_compat=_use_compat
    )
    return xp


def get_torch():
    from array_api_compat import torch

    return torch


def get_cupy():
    from array_api_compat import cupy

    return cupy


def get_numpy():
    from array_api_compat import numpy

    return numpy


if TYPE_CHECKING:
    import array_api_strict as xp
    from array_api_strict._array_object import Array as ArrayAPIArray
    from array_api_strict._array_object import Device as ArrayAPIDevice
    from array_api_strict._array_object import Dtype as ArrayAPIDtype

    Array: TypeAlias = ArrayAPIArray
    Device: TypeAlias = ArrayAPIDevice
    Dtype: TypeAlias = ArrayAPIDtype

    array_api_module: TypeAlias = xp


def numpy_only_compatibility(numpy_func):
    """
    Apply this decorator to numpy only functions to make them compliant
    with the array-api
    numpy_func: Callable
        signature (*args, **kwargs) -> array like object
    """

    @functools.wraps(numpy_func)
    def func(*args, **kwargs) -> "Array":
        array_args = list(filter(is_array_api_obj, args))
        array_kwargs = list(filter(is_array_api_obj, kwargs.values()))
        xp = array_namespace(*array_args, *array_kwargs)
        numpy_args = []
        for arg in args:
            arg_ = arg
            if is_array_api_obj(arg):
                arg_ = _to_numpy(arg)
            numpy_args.append(arg_)
        numpy_kwargs = {}
        for k, arg in kwargs.items():
            arg_ = arg
            if is_array_api_obj(arg):
                arg_ = _to_numpy(arg)
            numpy_kwargs[k] = arg_

        try:
            devices = set(map(get_device, array_args + array_kwargs))
            if len(devices) != 1:
                raise TypeError(f"Multiple devices found in args: {devices}")
            (device,) = devices
        except TypeError:
            device = get_device(array_args[0])

        return xp.asarray(numpy_func(*numpy_args, **numpy_kwargs), device=device)

    return func


@overload
def to_numpy(x: Any) -> numpy.ndarray: ...
@overload
def to_numpy(*xs: Any) -> tuple[numpy.ndarray, ...]: ...
def to_numpy(*xs):
    ret = tuple([_to_numpy(x) for x in xs])
    if len(xs) == 1:
        return ret[0]
    else:
        return ret


def _to_numpy(x: "Array") -> numpy.ndarray:
    if is_numpy_array(x):
        return x
    elif is_cupy_array(x):
        return x.get()
    elif is_torch_array(x):
        return x.cpu().numpy()
    else:
        return numpy.asarray(x, copy=True, device="cpu")


def _is_numpy_namespace(xp):
    return xp.__name__ in ("numpy", "array_api_compat.numpy")


def _is_cupy_namespace(xp):
    return xp.__name__ in ("cupy", "array_api_compat.cupy")


def _is_torch_namespace(xp):
    return xp.__name__ in ("torch", "array_api_compat.torch")


def get_prefered_namespace_device(
    xp: "array_api_module | None" = None,  # type: ignore
    device: "Device | None" = None,
    gpu: bool | None = None,
) -> "tuple[array_api_module, Device]":  # type: ignore
    if xp is not None:
        if device is not None:
            return xp, device
        if _is_torch_namespace(xp):
            xp = get_torch()
            if gpu is None:
                device = "cuda" if spfluo.has_torch_cuda() else "cpu"
            elif gpu:
                device = "cuda"
            else:
                device = "cpu"
        elif _is_cupy_namespace(xp):
            xp = get_cupy()
            if gpu is False:
                raise RuntimeError(f"{xp} cannot create non cuda arrays")
            device = None
        elif _is_numpy_namespace(xp):
            xp = get_numpy()
            if gpu is True:
                raise RuntimeError(f"{xp} cannot create gpu arrays")
            device = "cpu"
        else:
            raise RuntimeError(
                f"{xp} not supported. Must be one of torch, cupy or {numpy}"
            )
    else:
        if device is not None:
            raise RuntimeError("device provided but xp not provided.")
        if gpu is None:
            if spfluo.has_torch_cuda():
                xp = get_torch()
                device = "cuda"
            elif spfluo.has_cupy():
                xp = get_cupy()
                device = None
            elif spfluo.has_torch():
                xp = get_torch()
                device = "cpu"
            else:
                xp = numpy
                device = "cpu"
        elif gpu:
            if spfluo.has_torch_cuda():
                xp = get_torch()
                device = "cuda"
            elif spfluo.has_cupy():
                xp = get_cupy()
                device = None
            else:
                raise RuntimeError("GPU asked but no backend found")
        else:
            if spfluo.has_torch():
                xp = get_torch()
                device = "cpu"
            else:
                xp = numpy
                device = "cpu"
    return xp, device


__all__ = [array_namespace, is_array_api_obj, to_device, to_numpy, get_device]

if TYPE_CHECKING:
    __all__ = [
        Array,
        array_namespace,
        is_array_api_obj,
        to_device,
        to_numpy,
        array_api_module,
        get_device,
    ]
