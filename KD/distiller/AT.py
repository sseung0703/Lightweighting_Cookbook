import jax
import jax.numpy as jnp
from jax import tree_util

from flax.training import common_utils
import optax

'''
    Knowledge is defined by attention map which is L2-norm of each feature point.
    
    @misc{https://doi.org/10.48550/arxiv.1612.03928,
        doi = {10.48550/ARXIV.1612.03928},
        url = {https://arxiv.org/abs/1612.03928},
        author = {Zagoruyko, Sergey and Komodakis, Nikos},
        keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
        title = {Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer},
        publisher = {arXiv},
        year = {2016},
        copyright = {arXiv.org perpetual, non-exclusive license}
    }
'''

keep_feats = ['stage/last']

def at(x):
    y = jnp.reshape(jnp.mean(x**2, -1), [x.shape[0], -1])
    y = y / jnp.maximum(jnp.linalg.norm(y, axis = 1, keepdims=True), 1e-3)
    return y

def objective(logits, student_feats, teacher_feats, label, beta = 1e3):
    """
       Objective function to train student network with teacher knowledge.

       Args:
           logits: output of the student network. This repository assume that task of neural networks is classification.
           label : label data
           student_feats: student feature maps or vectors that expressed above.
           teacher_feats: teacher feature maps or vectors that expressed above.

       Return:
           loss : cross-entropy loss + distillation loss

    """
    one_hot_labels = common_utils.onehot(label, num_classes=logits.shape[-1])
    loss = jnp.mean( optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels) )

    if student_feats is not None:
        at_loss = []
        for sf, tf in zip(tree_util.tree_leaves(student_feats), tree_util.tree_leaves(teacher_feats)):
            at_loss.append( jnp.mean((at(sf) - at(tf))**2 ))
        loss = loss + sum(at_loss) * beta
    return loss


