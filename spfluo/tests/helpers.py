import spfluo
from spfluo.utils.array import numpy as np

testing_libs = [(np, None)]
ids = ["numpy"]

if spfluo.has_cupy:
    from spfluo.utils.array import cupy

    testing_libs.append((cupy, None))
    ids.append("cupy")
if spfluo.has_torch:
    from spfluo.utils.array import torch

    testing_libs.append((torch, "cpu"))
    ids.append("torch-cpu")
    if spfluo.has_torch_cuda:
        testing_libs.append((torch, "cuda"))
        ids.append("torch-cuda")
