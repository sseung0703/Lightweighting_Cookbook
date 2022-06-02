import os

import jax
import jax.numpy as jnp
from jax import tree_util

from flax import jax_utils
import utils

def prune(state, datasets, frr, ori_flops):
    """
        Collect filter importance only one time and prune the network at once.
        There is only one pruning iteration, so pruning cost is much lower than others.
        This strategy looks too naive, but lots of algorithms, e.g., FPGM, Hrank, have adopted this strategy.

        Args:
            state:
            datasets:
            frr:
            ori_flops:

        return:
            state: state that contains pruned model, params, and batch_stats.
                   Note that, at this point, pruning is simulated by masking and "actual pruning" should be done.

    """
    ## Gather importance of each mask
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    variables = jax_utils.unreplicate(variables)

    input_size = (1, *datasets.input_size)
    dummy_input = jnp.ones(input_size, datasets.dtype)

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
        cur_flops, cur_n_params = utils.profile_model('#%s Pruned model'%(str(step).rjust(3)), datasets.input_size, state, datasets.dtype, log = False)
        
        if cur_flops / ori_flops < 1 - frr:
            for fine_step in range(1, step_per_filters):
                th = th_list[ (step-1) * step_per_filters + fine_step ]

                new_mask_params = {k: imp2mask(p, imp, th) for (k,p), (k2, imp) in zip(mask_params.items(), importance.items())}
                state = state.replace(params = state.params.copy(new_mask_params))
                cur_flops, cur_n_params = utils.profile_model('#%s Pruned model'%(str(step).rjust(3)), datasets.input_size, state, datasets.dtype, log = False)

                if cur_flops / ori_flops < 1 - frr:
                    th = th_list[ (step-1) * step_per_filters + fine_step - 1 ]

                    new_mask_params = {k: imp2mask(p, imp, th) for (k,p), (k2, imp) in zip(mask_params.items(), importance.items())}
                    state = state.replace(params = state.params.copy(new_mask_params))
                    cur_flops, cur_n_params = utils.profile_model('Pruned model', datasets.input_size, state, datasets.dtype)
                    return state 

