import importlib.util


def has_cupy():
    return importlib.util.find_spec("cupy") is not None


def has_torch():
    return importlib.util.find_spec("torch") is not None


def has_torch_cuda():
    if has_torch():
        import torch

        if torch.cuda.is_available():
            return True
    return False
