from typing import Any

import jax
import jax.numpy as jnp
from jax import tree_util

import flax
from flax import jax_utils
import flax.linen as nn

def measure(name, layer, x, y):
    """
        This criterion defines the filter norm as its importance.

        Args:
            name: name of layer to check whether importance should be collected at the given layer or not.
            layer: layer of the network
            x: input data of the layer
            y: output data of the layer

        Returns:
            importance: measured importance. if given layer is not utilized to measure importance, return None
    """
    if 'conv' in name:
        kernel = layer.variables['params']['kernel']
        Do = kernel.shape[-1]
        importance = jnp.linalg.norm(jnp.reshape(kernel, [-1, Do]), axis = 0)
    else:
        importance = None
    return importance

def collect_importance(state, datasets):
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    variables = jax_utils.unreplicate(variables)

    input_size = (1, *datasets.input_size)
    dummy_input = jnp.ones(input_size, datasets.dtype)

    _, new_state = state.apply_fn(variables, dummy_input, mutable = ['batch_stats','importance'])
    importance = new_state['importance']
    importance = {k: sum([i for i in imp['importance'] if i is not None]) for k,imp in importance.items()}
    return importance
