import os

import jax
import jax.numpy as jnp
from jax import tree_util

from flax import jax_utils
import utils

def prune(model, state, datasets, frr, ori_flops, ori_n_params):
    """
        Create main training step.
        This function is based on the below flax example.
        https://github.com/google/flax/blob/main/examples/imagenet/train.py

        Args:
            weight_decay: l2 regularization strength.

        return:
            train_step: training step for given model and trainable variables. whole computations are jit compiled and pmapped.
            sync_batch_stats: synchronize each batch stats on multiple devices.

    """
    ## Gather importance of each mask
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    variables = jax_utils.unreplicate(variables)

    input_size = (1, *datasets.input_size)
    dummy_input = jnp.ones(input_size, model.dtype)

    _, new_state = state.apply_fn(variables, dummy_input, mutable = ['batch_stats','importance'])
    importance = new_state['importance']
    importance = {k: sum([i for i in imp['importance'] if i is not None]) for k,imp in importance.items()}
    
    th_list = jnp.sort(jnp.concatenate(tree_util.tree_leaves(importance), 0))

    mask_params = {k:p for k,p in state.params.items() if 'mask' in k}

    step_per_filters = th_list.shape[0]//25
    for step in range(1, 25):
        th = th_list[step * step_per_filters] 

        def imp2mask(p, imp, th):
            mask = jnp.logical_or(imp > th, imp == jnp.max(imp)).astype(jnp.float32)
            mask = jnp.stack([mask] * p['mask'].shape[0])
            p = p.copy({'mask': mask})
            return p

        new_mask_params = {k: imp2mask(p, imp, th) for (k,p), (k2, imp) in zip(mask_params.items(), importance.items())}
        state = state.replace(params = state.params.copy(new_mask_params))
        cur_flops, cur_n_params = utils.profile_model('#%s Pruned model'%(str(step).rjust(3)), datasets.input_size, state, model.dtype, log = False)
        
        if cur_flops / ori_flops < 1 - frr:
            for fine_step in range(1, step_per_filters):
                th = th_list[ (step-1) * step_per_filters + fine_step ]

                new_mask_params = {k: imp2mask(p, imp, th) for (k,p), (k2, imp) in zip(mask_params.items(), importance.items())}
                state = state.replace(params = state.params.copy(new_mask_params))
                cur_flops, cur_n_params = utils.profile_model('#%s Pruned model'%(str(step).rjust(3)), datasets.input_size, state, model.dtype, log = False)

                if cur_flops / ori_flops < 1 - frr:
                    th = th_list[ (step-1) * step_per_filters + fine_step - 1 ]

                    new_mask_params = {k: imp2mask(p, imp, th) for (k,p), (k2, imp) in zip(mask_params.items(), importance.items())}
                    state = state.replace(params = state.params.copy(new_mask_params))
                    cur_flops, cur_n_params = utils.profile_model('Pruned model', datasets.input_size, state, model.dtype)
                    return state 

