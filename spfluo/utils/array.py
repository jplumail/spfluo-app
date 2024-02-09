import functools
from typing import TYPE_CHECKING, TypeAlias

from numpy.array_api._array_object import Array as ArrayAPIArray

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


__all__ = [Array, array_namespace, is_array_api_obj, to_device, *libs]

if TYPE_CHECKING:
    __all__ = [
        Array,
        array_namespace,
        is_array_api_obj,
        to_device,
        *libs,
        array_api_module,
    ]
