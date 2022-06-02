#  Filter Pruning Cookbook in JAX/FLAX
## Introduction
- Network pruning is attracting a lot of attention as a lightweight technique to reduce computation and memory cost by directly removing parameters of deep neural networks (DNNs).
- Filter pruning is advantageous in accelerating using the basic linear algebra subprograms (BLAS) library because it eliminates parameters in units of filters.
- This sub-project aims to implement various filter pruning algorithms in a single framework to make it easy to compare and mix them.
- Here, this repository assumes that the pre-trained network is given.
- **Currently, this repository work fine, but its readability has to be improved.**

## How to use
```
  python train_w_fp.py --gpu_id 2 \
    --trained_param /path/to/network/be/pruned \
    --distiller FilterNorm --transfer AtOnce \
    --dataset CIFAR10 --train_path ~/test
```

### AtOnce transfer
<p align="center">

|   Methods   |      Accuracy |
|:-----------:|:-------------:|
|   Baseline  |         93.15 |
</p>

## Implemented Filter pruning Methods
