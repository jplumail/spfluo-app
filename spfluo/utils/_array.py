from typing import TypeVar

import array_api_compat
from cupy import ndarray as cupy_array
from numpy import ndarray as numpy_array
from torch import Tensor as torch_array

Array = TypeVar("Array", numpy_array, cupy_array, torch_array)


def cpu_only_compatibility(cpu_func):
    """
    Apply this decorator to cpu only functions to make them compliant with the array-api
    cpu_func: Callable
        signature (*args, **kwargs) -> array like object
    """
    is_array = array_api_compat.is_array_api_obj

    def func(*args, **kwargs) -> Array:
        array_args = list(filter(is_array, args))
        array_kwargs = list(filter(is_array, kwargs.values()))
        xp = array_api_compat.array_namespace(*array_args, *array_kwargs)
        cpu_args = []
        for arg in args:
            arg_ = arg
            if is_array(arg):
                arg_ = xp.asarray(arg)
                arg_ = xp.to_device(arg_, "cpu")
            cpu_args.append(arg_)
        cpu_kwargs = {}
        for k, arg in kwargs.items():
            arg_ = arg
            if is_array(arg):
                arg_ = xp.asarray(arg)
                arg_ = xp.to_device(arg_, "cpu")
            cpu_kwargs[k] = arg_

        devices = set(map(xp.device, array_args + array_kwargs))
        if len(devices) != 1:
            raise TypeError(f"Multiple devices found in args: {devices}")
        (device,) = devices

        return xp.asarray(cpu_func(*cpu_args, **cpu_kwargs), device=device)

    return func
