from typing import Any
import os

import jax
import jax.numpy as jnp
from jax import tree_util

import flax
import flax.linen as nn
from flax import jax_utils
from flax.core.frozen_dict import FrozenDict

from nets import ResNet

def MaskInitialization(arch, model):
    mask_dict = {}
    if 'ResNet' in arch:
        mask_dict['conv_init/out_mask'] = Mask(model.num_filters, name = 'conv_init/out_mask')
        group_mask = mask_dict['conv_init/out_mask']

        num_filters_in = model.num_filters

        for i, block_size in enumerate(model.stage_sizes):
            num_filters = model.num_filters * 2 ** i
            for j in range(block_size):
                if model.block_cls == ResNet.ResNetBlock: 
                    mask_dict['block_%d_%d/conv0/in_mask'%(i,j)] = group_mask 
                    mask_dict['block_%d_%d/conv0/out_mask'%(i,j)] = Mask(num_filters, name = 'block_%d_%d/conv0/out_mask'%(i,j))
                    in_mask = mask_dict['block_%d_%d/conv0/out_mask'%(i,j)]

                    if (i > 0 and j == 0) or num_filters != num_filters_in:
                        mask_dict['block_%d_%d/conv_proj/in_mask'%(i,j)] = group_mask 
                        mask_dict['block_%d_%d/conv_proj/out_mask'%(i,j)] = Mask(num_filters, name = 'block_%d_%d/conv_proj/out_mask'%(i,j))
                        group_mask = mask_dict['block_%d_%d/conv_proj/out_mask'%(i,j)]

                    mask_dict['block_%d_%d/conv1/in_mask'%(i,j)] = in_mask 
                    mask_dict['block_%d_%d/conv1/out_mask'%(i,j)] = group_mask
                num_filters_in = num_filters

        mask_dict['classifier/in_mask'] = group_mask

        for k, mask in list(mask_dict.items()):
            if 'conv' in k and 'out_mask' in k:
                mask_dict[k.replace('conv','bn')] = mask
    return mask_dict

class Mask(nn.Module):
    """
        A linear transformation applied over the last dimension of the input.

        Attributes:
            features: the number of output features.
            use_bias: whether to add a bias to the output (default: True).
            dtype: the dtype of the computation (default: infer from input and params).
            param_dtype: the dtype passed to parameter initializers (default: float32).
            precision: numerical precision of the computation see `jax.lax.Precision`
            for details.
    """
    features: int
    dtype: Any = None
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        """
            Applies a linear transformation to the inputs along the last dimension.

            Returns:
                The transformed input.
        """
        mask = self.param('mask',
                          nn.initializers.ones,
                          (self.features),
                          self.param_dtype)
        return jnp.reshape(mask, (1,) * (x.ndim - 1) + (-1,))

def actual_pruning(arch, model, input_size, state):
    '''
        prune the network actually.
        currently, this function works well but looks not fancy, so it will be improve ASAP.
    '''
    state = jax_utils.unreplicate(state)
    variables = {'params': state.params, 'batch_stats': state.batch_stats}

    input_size = (1, *input_size)
    dummy_input = jnp.ones(input_size, model.dtype)

    _, new_state = state.apply_fn(variables, dummy_input, train = False, mutable = ['in_mask', 'out_mask'])
    in_mask = new_state['in_mask']
    out_mask = new_state['out_mask']
 
    features_dict = {}

    def pruning(in_mask, out_mask, keys, variables):
        for k in keys:
            in_mask = in_mask.get(k)
            out_mask = out_mask.get(k)
        
        in_mask = in_mask.get('in_mask', None)
        out_mask = out_mask.get('out_mask', None)

        if out_mask is not None:
            features_dict[keys[-1]] = int(jnp.sum(out_mask))

        pruned_variables = {}
        for k, v in variables.items():
            if 'conv' in keys[-1]:
                if 'kernel' in k:
                    if in_mask is not None:
                        v = v[:,:,in_mask.reshape(-1).astype(bool)]

                    if out_mask is not None:
                        v = v[:,:,:,out_mask.reshape(-1).astype(bool)]
                if 'bias' in k:
                    if out_mask is not None:
                        v = v[out_mask.reshape(-1).astype(bool)]

            if 'bn' in keys[-1]:
                if out_mask is not None:
                    v = v[out_mask.reshape(-1).astype(bool)]

            if 'classifier' in keys[-1]:
                if 'kernel' in k:
                    if in_mask is not None:
                        v = v[in_mask.reshape(-1).astype(bool)]

                    if out_mask is not None:
                        v = v[:,out_mask.reshape(-1).astype(bool)]

                if 'bias' in k:
                    if out_mask is not None:
                        v = v[out_mask.reshape(-1).astype(bool)]
            pruned_variables[k] = v

        return pruned_variables

    def rebuild_tree(frozen_dict, K):
        new_frozen_dict = {}
        for k, layer in frozen_dict.items():
            if 'mask' not in k:
                if any([ not(isinstance(p, FrozenDict) or isinstance(p, dict)) for _, p in layer.items()]) and layer:
                    new_frozen_dict[k] = pruning(in_mask, out_mask, K+[k], layer)
                else:
                    new_frozen_dict[k] = rebuild_tree(layer, K+[k])
        return new_frozen_dict

    model.mask_dict = {}
    model.features_dict = features_dict
    return model, rebuild_tree(state.params, []), rebuild_tree(state.batch_stats, [])
