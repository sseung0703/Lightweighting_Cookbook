# Knowledge distillation cookbook in JAX/FLAX
## Introduction
- Knowledge distillation (KD) is one of the solutions to build light-weighted CNNs.
  The main function of KD is to create and deliver a certain knowledge so that a student network behaves similarly to a teacher network. Since KD can be applied to various machine learning areas such as semi-supervised learning and zero-shot learning, KD has been received a lot of attention recently.
- Conventional KD algorithms defined information from several locations of CNN, e.g., intermediate feature maps and embedded feature vectors at the output end of CNN as the knowledge of CNN.
- Knowledge can be dilivered to the student network by three strategies, i.e., offline, online and self-distillation.

## How to use
### Offline transfer
```
  python train_w_kd.py --gpu_id 2 \
    --student_arch ResNet32 \
    --teacher_arch ResNet56 --teacher_param ../pretrained/res56_c10 \
    --distiller SoftLogits --transfer Offline \
    --dataset CIFAR10 --train_path ~/test
```
## Experimental Results
- I have used ResNet-56 and ResNet-32 as the teacher and the student network, respectively.
- All the algorithm is trained in the sample configuration, which is described in "distiller/*". I tried only several times to get acceptable performance, which means that my experimental results are perhaps not optimal.
- Although some of the algorithms used soft-logits parallelly in the paper, I used only the proposed knowledge distillation algorithm to make a fair comparison.

<p align="center">

|   Methods   |      Accuracy |
|:-----------:|:-------------:|
|   Teacher   |         93.96 |
|   Student   |         93.15 |
| Soft_logits |         93.41 |
|      AT     |         93.74 |
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/26036843/170808252-391a98c7-699b-456e-b758-da0a49ec30f7.jpeg" width="400">

</p>

## Implemented Knowledge Distillation Methods
- Soft-logit : The first knowledge distillation method for deep neural network. Knowledge is defined by softened logits. Because it is easy to handle it, many applied methods were proposed using it such as semi-supervised learning, defencing adversarial attack and so on.
  - [Geoffrey Hinton, et al. Distilling the knowledge in a neural network. arXiv:1503.02531, 2015.](https://arxiv.org/abs/1503.02531)

- Attention transfer (AT) : Knowledge is defined by attention map which is L2-norm of each feature point.
  - [Zagoruyko, Sergey et. al. Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer. arXiv preprint arXiv:1612.03928, 2016.](https://arxiv.org/pdf/1612.03928.pdf) [[the original project link](https://github.com/szagoruyko/attention-transfer)]

