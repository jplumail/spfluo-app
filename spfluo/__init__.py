try:
    import cupy  # noqa: F401

    has_cupy = True
except ImportError:
    has_cupy = False

try:
    import torch

    has_torch = True
    if torch.cuda.is_available():
        has_torch_cuda = True
    else:
        has_torch_cuda = False
except ImportError:
    has_torch = False
    has_torch_cuda = False
