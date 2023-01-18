# SPFluo_stage_reconstruction_symmetryC

This repository contains code for picking and single particle reconstruction in fluorescence imaging.

Pipeline :
1. [picking](spfluo/picking)
2. for centrioles only : [cleaning with segmentation](spfluo/segmentation) and [alignement](spfluo/alignement)
3. [reconstruction ab-initio](spfluo/reconstruction-ab-initio/)
4. [refinement](spfluo/refinement/)

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