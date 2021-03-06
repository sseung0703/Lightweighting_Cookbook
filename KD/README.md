# Knowledge distillation cookbook in JAX/FLAX
## Introduction
- Knowledge distillation (KD) is one of the solutions to build light-weighted CNNs.
  The main function of KD is to create and deliver a certain knowledge so that a student network behaves similarly to a teacher network. Since KD can be applied to various machine learning areas such as semi-supervised learning and zero-shot learning, KD has been received a lot of attention recently.
- Conventional KD algorithms defined information from several locations of CNN, e.g., intermediate feature maps and embedded feature vectors at the output end of CNN as the knowledge of CNN.
- Knowledge can be dilivered to the student network by three strategies, i.e., offline, online and self-distillation.

## How to use
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
- Note that, the current neumerical values are results of only one trial, so there will be quite large variance. They will be improved when I take GCP credits.

### Offline transfer
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

### Online transfer
<p align="center">

|   Methods   |      Accuracy |
|:-----------:|:-------------:|
|   Teacher   |         93.96 |
|   Student   |         93.15 |
| Soft_logits |         93.74 |
|      AT     |         93.67 |
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/26036843/170950164-c9b5fb95-929d-4cef-b97e-740e761a26e3.png" width="400">

</p>

## Implemented Knowledge Distillation Methods
- Knowledge distillation algorithms distill various information from various sensing points.
- In the points of distilled information, it is hard to concretely categorize previous methods, but to my knowledge, I suggest the three categories, i.e., activation, embedding process, and inter-data relation.
  - Activation: compare the student and teacher features without further data-level interaction. For example, AB compares each feature map activation boundaries.
  - embedding process: extract information from the embedding process by analyzing how each feature map is transformed in each stage. For example, FSP computes the Gram matrix of two feature maps and defines it as a flow of solving a problem.
  - Inter-data relation: build graph structure formed by inter-data relation. For example, RKD defines distance and angle as edge features.
  - Mixed knowledge: Some algorithms build a combination of several types of knowledge at physical or chemical levels. For example, MHGD utilizes knowledge of dataset embedding process, which cares about not only single data's embedding process but also inter-data relation.

- In the points of feature sensing position, there are two options, i.e., network output (a.k.a. embedded feature or response) and latent feature maps.
- Each feature has different characteristics, so you should choose it as what you want to achieve.

### Output (a.k.a. embedded feature or response) distillation:
  - Simple and efficient distillation schemes.
  - Gives minimal guidance to the student network. In other words, its constraints are usually weak to achieve SOTA performance but rarely cause the over-constraints problem.

- Soft-logits: The first knowledge distillation method for a deep neural network. Knowledge is defined by softened logits. Because it is easy to handle, many applied methods were proposed using it, such as semi-supervised learning, defencing adversarial attack, and so on.
  - [Geoffrey Hinton, et al. Distilling the knowledge in a neural network. arXiv:1503.02531, 2015.](https://arxiv.org/abs/1503.02531)

### Latent feature distillation:
  - Extract much more rich guidance to student networks using multiple latent feature maps.
  - Most SOTA algorithm has taken this strategy, but it is prone to give over-constraints that makes the student network hard to focus on its own task.

- Attention transfer (AT): Knowledge is defined by an attention map which is the L2-norm of each feature point.
  - [Zagoruyko, Sergey et. al. Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer. arXiv preprint arXiv:1612.03928, 2016.](https://arxiv.org/pdf/1612.03928.pdf) [[the original project link](https://github.com/szagoruyko/attention-transfer)]

## Implemented Transfer Methods
- Knowledge can be transferred by three kinds of strategies, i.e., offline, online, and self-distillation.
  - Offline distillation: Utilize the pre-trained teacher network, and it is frozen and used for extracting knowledge. The knowledge is not changed during the training framework and gives concrete guidance to the student network.
  - Online distillation: The teacher network is randomly initialized and trained with the student network simultaneously. At the initial points, information about the teacher network is not so hard to train for the student network, so the online learning strategy gives gentle guidance.
  - Self-distillation: In the case of self-distillation, there is no teacher network, so each algorithm has its own strategies to build pseudo teacher knowledge.
