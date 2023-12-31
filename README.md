# PoissonNet

### [**Paper**](https://arxiv.org/abs/2308.01766) | [**Project Page**](https://github.com/arsenal9971/PoissonNet/)

![](./media/figure_benchmark.png)

This repository contains the implementation of the paper:

PoissonNet: Resolution-Agnostic 3D Shape Reconstruction using Fourier Neural Operators 
Anonymous

We are currently working on a cleaned-up version of this code that includes more documentation and pre-trained weights. If you find our code or paper useful, please consider citing
```bibtex
@article{anonymous,
  title={PoissonNet: Resolution-Agnostic 3D Shape Reconstruction using Fourier Neural Operators},
  author={Anonymous},
  journal={arXiv preprint},
  year={2023}
}
```


## Installation

If you want to use our pre-trained model, you can import it
```
PoissonNet = torch.hub.load('arsenal9971/PoissonNet', '')
```

You need to first install all the dependencies. For that you can use [anaconda](https://www.anaconda.com/). 

You can create an anaconda environment called `poissonnet` using
```
conda env create -f environment.yaml
conda activate poissonnet
```

## Training - Quick Start

First, run the script to get the demo data:

```bash
bash scripts/download_data.sh
```
