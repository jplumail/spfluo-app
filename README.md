### spfluo-app

This is the main repository of the `spfluo-app`.

`spfluo-app` aims at bringing 3D reconstruction techniques to fluorescence imaging!

`spfluo-app` is built on top of a fork [Scipion](https://scipion.i2pc.es/), so that it runs on Windows.

## Install

Go to `spfluo-app` [installation page](https://spfluo.icube.unistra.fr/en/usage/installation.html) and follow the instructions. 

## Install for developement

Run with uv:
```sh
git clone https://github.com/jplumail/spfluo-app
cd spfluo-app
uv run --extra cu124 spfluo-app  # run with cuda 12.4 torch
uv run --extra cpu spfluo-app  # run with cpu torch
```