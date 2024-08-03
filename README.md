# SP-Fluo

This repository contains code for picking and single particle reconstruction in fluorescence imaging.

Pipeline :
1. [picking](spfluo/picking)
2. for centrioles only : [cleaning with segmentation](spfluo/segmentation) and [alignement](spfluo/alignement)
3. [reconstruction ab-initio](spfluo/ab_initio_reconstruction/)
4. [refinement](spfluo/refinement/)

## Installation

```bash
git clone https://github.com/jplumail/spfluo
```

```bash
cd spfluo
pip install .
```

## Run tests

To test for python 3.9, 3.10 and 3.11:
```bash
hatch test spfluo
```

For python 3.8:
```bash
hatch env run -e hatch-test-38 pytest -- spfluo
```