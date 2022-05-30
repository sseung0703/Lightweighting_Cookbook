import jax
import jax.numpy as jnp
from jax import tree_util

import flax
import flax.linen as nn

def GateInitialization(arch, model):
    if 'ResNet' in arch:
        n_layers = int(arch[6:])

        if n_layer in [18, 20, 32, 34, 56]:
            ## ResNet-family with Basicblock
            


        elif n_layer in [50, 101, 152, 200]:
            ## ResNet-family with BottleNectblock
