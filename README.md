# SPFluo_stage_reconstruction_symmetryC

This repository contains code for picking and single particle reconstruction in fluorescence imaging.

Pipeline :
1. [picking](code/picking)
2. for centrioles only : [cleaning with segmentation](code/segmentation) and [alignement](code/alignement)
3. [reconstruction ab-initio](code/reconstruction-ab-initio/)

## Installation
```bash
git clone https://github.com/dfortun2/SPFluo_stage_reconstruction_symmetryC
```

```bash
cd SPFluo_stage_reconstruction_symmetryC
pip install .
```

### Optional dependencies

- [Pytorch](https://pytorch.org/)
- [Cupy](https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-pypi)