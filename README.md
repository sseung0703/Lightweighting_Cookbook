# Training and Lightweighting Cookbook in JAX/FLAX 
This project page attempted to build neural network training and lightweighting cookbook including three kinds of lightweighting solutions, i.e., knowledge distillation, filter pruning, and quantization. <br>
It will be a quite long term project, so please get patient and keep watching this repository 🤗.

## Requirements
- jax
- flax
- tensorflow ( to download CIFAR dataset )

# Basic training framework in JAX/FLAX
## How to use
1. Move to the codebase.
2. Train and evaluate our model by the below command.

```
  # ResNet-56 on CIFAR10
  python train.py --gpu_id 0 --arch ResNet-56 --dataset CIFAR10 --train_path ../test
  python test.py --gpu_id 0 --arch ResNet-56 --dataset CIFAR10 --trained_param ../test/checkpoint_200
```

## Experimental comparison with other common deep learning libraries, i.e., Tensorflow2 and Pytorch
- Hardware: GTX 1080
- Tensorflow implementation [[link](https://github.com/sseung0703/EKG)]
- Pytorch implementation [[link](https://github.com/akamaster/pytorch_resnet_cifar10)]
- In order to check only training time except for model and data preparation, training time is calculated from the second to the last epoch.

- Note that Accuracy on CIFAR dataset has a quite large variance <br>
  so that you should focus on another metrics, i.e., training time.
- As you can notice, JAX and TF are much faster than Pytorch because of JIT compiling. This performance gap 

<p align="center">

| Library | Accuracy| Time (m)|
|:-------:|:-------:|:-------:|
| JAX     |   93.98 |      54 |
| TF      |   93.91 |      53 |
| Pytorch |   93.80 |      69 |

</p>

# TO DO
- [x] Basic training and test framework
  - [x] Dataprovider in JAX
  - [x] Naive training framework
  - [x] Monitoring by Tensorboard
  - [ ] Enlarge model zoo
  
- [ ] Knowledge distillation framework
  - [ ] Basic framework
  - [ ] Off-line distillation
  - [ ] On-line distillation
  - [ ] Self distillation
  - [ ] Enlarge the distillation algorithm zoo

- [ ] Filter pruning framework
  - [ ] Basic framework
  - [ ] Criterion-based pruning
  - [ ] Search-based pruning
  - [ ] Enlarge filter pruning algorithm zoo

- [ ] Quantization framework
  - [ ] Basic framework
  - [ ] Quantization aware training
  - [ ] Post Training Quantization
  - [ ] Enlarge quantization algorithm zoo

- [ ] Tools for handy usage.
