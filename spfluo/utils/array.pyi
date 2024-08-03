from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    import array_api_strict as xp
    from array_api_strict._array_object import Array as ArrayAPIArray
    from array_api_strict._array_object import Device as ArrayAPIDevice
    from array_api_strict._array_object import Dtype as ArrayAPIDtype

    Array: TypeAlias = ArrayAPIArray
    Device: TypeAlias = ArrayAPIDevice
    Dtype: TypeAlias = ArrayAPIDtype

    array_api_module: TypeAlias = xp

def array_namespace(*xs) -> "array_api_module": ...
def get_prefered_namespace_device(
    xp: "array_api_module | None" = None,  # type: ignore
    device: "Device | None" = None,
    gpu: bool | None = None,
) -> "tuple[array_api_module, Device | None]": ...  # type: ignore
def median(a: "Array", axis: int | None, xp: "array_api_module"): ...
