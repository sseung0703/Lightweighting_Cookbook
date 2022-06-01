from typing import Any

import jax
import jax.numpy as jnp
from jax import tree_util

import flax
import flax.linen as nn

def measure(name, layer, x, y):
    """
        Applies a linear transformation to the inputs along the last dimension.

        Returns:
            The transformed input.
    """
    if 'conv' in name:
        kernel = layer.variables['params']['kernel']
        Do = kernel.shape[-1]
        return jnp.linalg.norm(jnp.reshape(kernel, [-1, Do]), axis = 0)
    else:
        return 
