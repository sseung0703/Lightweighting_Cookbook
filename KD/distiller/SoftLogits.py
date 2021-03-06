import jax
import jax.numpy as jnp

from flax.training import common_utils
import optax

'''
    The first knowledge distillation method for deep neural network. Knowledge is defined by softened logits.
    Because it is easy to handle it, many applied methods were proposed using it such as semi-supervised learning, defencing adversarial attack and so on.
    
    @misc{https://doi.org/10.48550/arxiv.1503.02531,
        doi = {10.48550/ARXIV.1503.02531},
        url = {https://arxiv.org/abs/1503.02531},
        author = {Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff},
        keywords = {Machine Learning (stat.ML), Machine Learning (cs.LG), Neural and Evolutionary Computing (cs.NE), FOS: Computer and information sciences, FOS: Computer and information sciences},
        title = {Distilling the Knowledge in a Neural Network},
        publisher = {arXiv},
        year = {2015},
        copyright = {arXiv.org perpetual, non-exclusive license}
    }
'''

## Soft logits doesn't requires additional feature map.
keep_feats = ['classifier/out']

def kld(x, y, T = 1, axis = -1, keepdims=True):
    return T * jnp.sum(jax.nn.softmax(x/T, axis = axis)*(jax.nn.log_softmax(x/T, axis = axis) - jax.nn.log_softmax(y/T, axis = axis)), axis = axis, keepdims=keepdims)

def objective(logits, label, student_feats, teacher_feats, T = 4, alpha = 0.5):
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
        kld_loss = jnp.mean(kld(student_feats['classifier']['keep_feats'][0], teacher_feats['classifier']['keep_feats'][0], T = T))
        loss = loss * alpha + kld_loss * ( 1 - alpha)
    return loss


