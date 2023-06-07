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